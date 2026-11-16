/*
 * SD card over SPI (SPIM0) on the nRF52805, plus simple
 *             file logging through the FatFs library
 * Blocking
 */

#include <stdbool.h>
#include <string.h>
#include "nrf.h" //nrfx.h library download 
#include "sd.h"
#include "ff.h"        //have to download from library
#include "diskio.h"    



// Need to go through RAM variables
static uint8_t spi_tx_byte;
static uint8_t spi_rx_byte;

//GPIO for CS 
static inline void cs_low(void)  { NRF_P0->OUTCLR = (1UL << SD_PIN_CS); }
static inline void cs_high(void) { NRF_P0->OUTSET = (1UL << SD_PIN_CS); }



void sd_spi_config(uint32_t freq)
{
    //Can only change registers while SPIM disabled 
    NRF_SPIM0->ENABLE = SPIM_ENABLE_ENABLE_Disabled;

    /*GPIO set up 
     * SCK : output, idle level = CPOL (0 = low)
     * MOSI: output, idle low
     * MISO: input. Pull-up added because SD cards need MISO pulled
     *       high when they are not driving it.
     * CS  : plain output, idle HIGH (card not selected)            */
    NRF_P0->OUTCLR = (1UL << SD_PIN_SCK) | (1UL << SD_PIN_MOSI);
    NRF_P0->OUTSET = (1UL << SD_PIN_CS);

    NRF_P0->PIN_CNF[SD_PIN_SCK] =
          (GPIO_PIN_CNF_DIR_Output       << GPIO_PIN_CNF_DIR_Pos)
        | (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos);
    NRF_P0->PIN_CNF[SD_PIN_MOSI] =
          (GPIO_PIN_CNF_DIR_Output       << GPIO_PIN_CNF_DIR_Pos)
        | (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos);
    NRF_P0->PIN_CNF[SD_PIN_MISO] =
          (GPIO_PIN_CNF_DIR_Input        << GPIO_PIN_CNF_DIR_Pos)
        | (GPIO_PIN_CNF_INPUT_Connect    << GPIO_PIN_CNF_INPUT_Pos)
        | (GPIO_PIN_CNF_PULL_Pullup      << GPIO_PIN_CNF_PULL_Pos);
    NRF_P0->PIN_CNF[SD_PIN_CS] =
          (GPIO_PIN_CNF_DIR_Output       << GPIO_PIN_CNF_DIR_Pos)
        | (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos);


    NRF_SPIM0->PSEL.SCK  = SD_PIN_SCK;
    NRF_SPIM0->PSEL.MOSI = SD_PIN_MOSI;
    NRF_SPIM0->PSEL.MISO = SD_PIN_MISO;

    //Clock Speed 
    NRF_SPIM0->FREQUENCY = freq;

    //SPI Mode 0 
    NRF_SPIM0->CONFIG =
          (SPIM_CONFIG_ORDER_MsbFirst  << SPIM_CONFIG_ORDER_Pos)
        | (SPIM_CONFIG_CPHA_Leading    << SPIM_CONFIG_CPHA_Pos)
        | (SPIM_CONFIG_CPOL_ActiveHigh << SPIM_CONFIG_CPOL_Pos);


    NRF_SPIM0->ORC = 0xFF;

    NRF_SPIM0->ENABLE = SPIM_ENABLE_ENABLE_Enabled;
}


/* ---------------------------------------------------------------------
 * spi_transfer()
 * Sends tx_len bytes from 'tx' while receiving rx_len bytes into 'rx'.
 * Uses EasyDMA (PS 6.16.3):
 *   - if rx_len > tx_len, the extra bytes sent are ORC (0xFF)
 *   - if tx_len > rx_len, the extra bytes received are thrown away
 * Waits for the END event, which means both directions are done.
 * --------------------------------------------------------------------- */
static void spi_transfer(const uint8_t *tx, uint32_t tx_len,
                         uint8_t *rx, uint32_t rx_len)
{
    NRF_SPIM0->TXD.PTR    = (uint32_t)(tx ? tx : &spi_tx_byte);
    NRF_SPIM0->TXD.MAXCNT = tx_len;
    NRF_SPIM0->RXD.PTR    = (uint32_t)(rx ? rx : &spi_rx_byte);
    NRF_SPIM0->RXD.MAXCNT = rx_len;

    NRF_SPIM0->EVENTS_END = 0;
    NRF_SPIM0->TASKS_START = 1;
    while (NRF_SPIM0->EVENTS_END == 0) {}
    NRF_SPIM0->EVENTS_END = 0;
}

// Send one byte and return the byte received at the same time.
static uint8_t spi_byte(uint8_t out)
{
    spi_tx_byte = out;
    spi_transfer(&spi_tx_byte, 1, &spi_rx_byte, 1);
    return spi_rx_byte;
}



/* SD command numbers used here */
#define CMD0    0    /* GO_IDLE_STATE: reset, enter SPI mode           */
#define CMD8    8    /* SEND_IF_COND: check voltage / card version      */
#define CMD16   16   /* SET_BLOCKLEN: set block size to 512 (old cards) */
#define CMD17   17   /* READ_SINGLE_BLOCK                               */
#define CMD24   24   /* WRITE_BLOCK                                     */
#define CMD55   55   /* APP_CMD: "next command is an ACMD"              */
#define CMD58   58   /* READ_OCR: read card's capacity type             */
#define ACMD41  41   /* SD_SEND_OP_COND: start card initialisation      */


#define R1_IDLE         0x01
#define R1_ILLEGAL_CMD  0x04


#define TOKEN_START_BLOCK  0xFE   /* comes right before 512 data bytes */


#define TIMEOUT_INIT_LOOPS   2000    /* ACMD41 attempts (~1.5 s) */
#define TIMEOUT_TOKEN_LOOPS  100000  /* wait for data token      */
#define TIMEOUT_BUSY_LOOPS   500000  /* wait for write to finish */

static bool card_is_high_capacity;
static bool card_ready;

// Wait until the card stops holding MISO low (= no longer busy)
static bool wait_not_busy(uint32_t loops)
{
    while (loops--) {
        if (spi_byte(0xFF) == 0xFF) {
            return true;
        }
    }
    return false;
}


static void card_deselect(void)
{
    cs_high();
    spi_byte(0xFF);
}

//Make sure card ready
static bool card_select(void)
{
    cs_low();
    spi_byte(0xFF);
    if (wait_not_busy(TIMEOUT_BUSY_LOOPS)) {
        return true;
    }
    card_deselect();
    return false;
}

/* ---------------------------------------------------------------------
 * sd_command()
 * Sends one 6-byte SD command and returns the 1-byte R1 response.
 *
 *   byte 0   : 0x40 | command number
 *   bytes 1-4: 32-bit argument, most significant byte first
 *   byte 5   : CRC7 << 1 | 1   (only checked for CMD0 and CMD8 in
 *              SPI mode, so we hard-code those two)
 *
 * The card answers within 1-8 bytes; a valid answer has bit 7 = 0.
 * The card is left SELECTED so the caller can read/write data next.
 * --------------------------------------------------------------------- */
static uint8_t sd_command(uint8_t cmd, uint32_t arg)
{
    if (cmd == ACMD41) {
        uint8_t r = sd_command(CMD55, 0);
        if (r > R1_IDLE) {
            return r;
        }
    }

    card_deselect();
    if (!card_select()) {
        return 0xFF;
    }

    static uint8_t frame[6];          /* static: must be in RAM for DMA */
    frame[0] = 0x40 | cmd;
    frame[1] = (uint8_t)(arg >> 24);
    frame[2] = (uint8_t)(arg >> 16);
    frame[3] = (uint8_t)(arg >> 8);
    frame[4] = (uint8_t)(arg);
    frame[5] = (cmd == CMD0) ? 0x95 :   /* valid CRC for CMD0, arg 0     */
               (cmd == CMD8) ? 0x87 :   /* valid CRC for CMD8, arg 0x1AA */
                               0x01;    /* CRC ignored; stop bit = 1     */
    spi_transfer(frame, 6, NULL, 0);

    /* Poll for the response (bit 7 clear). */
    uint8_t r1;
    for (int i = 0; i < 10; i++) {
        r1 = spi_byte(0xFF);
        if ((r1 & 0x80) == 0) {
            break;
        }
    }
    return r1;
}


int sd_init(void)
{
    static uint8_t resp[4];
    uint8_t r1;

    card_ready = false;

    sd_spi_config(SPIM_FREQUENCY_FREQUENCY_K250);


    cs_high();
    for (int i = 0; i < 10; i++) {
        spi_byte(0xFF);
    }


    r1 = sd_command(CMD0, 0);
    if (r1 != R1_IDLE) {
        card_deselect();
        return -1;                       /* no card, or not responding */
    }

    bool v2_card = false;
    r1 = sd_command(CMD8, 0x000001AA);
    if (r1 == R1_IDLE) {
        spi_transfer(NULL, 0, resp, 4);  
        if (resp[2] != 0x01 || resp[3] != 0xAA) {
            card_deselect();
            return -2;                 
        }
        v2_card = true;
    } else if (!(r1 & R1_ILLEGAL_CMD)) {
        card_deselect();
        return -3;
    }


    uint32_t acmd41_arg = v2_card ? 0x40000000 : 0;
    int tries = TIMEOUT_INIT_LOOPS;
    do {
        r1 = sd_command(ACMD41, acmd41_arg);
        for (volatile int d = 0; d < 1000; d++) { }   
    } while (r1 != 0x00 && --tries > 0);
    if (r1 != 0x00) {
        card_deselect();
        return -4;                       
    }


    card_is_high_capacity = false;
    if (v2_card) {
        if (sd_command(CMD58, 0) != 0x00) {
            card_deselect();
            return -5;
        }
        spi_transfer(NULL, 0, resp, 4);
        card_is_high_capacity = (resp[0] & 0x40) != 0;
    }
    if (!card_is_high_capacity) {
        if (sd_command(CMD16, 512) != 0x00) {
            card_deselect();
            return -6;
        }
    }
    card_deselect();


    sd_spi_config(SPIM_FREQUENCY_FREQUENCY_M8);

    card_ready = true;
    return 0;
}


int sd_read_block(uint32_t block, uint8_t *buf)
{
    static uint8_t crc[2];
    uint32_t addr = card_is_high_capacity ? block : block * 512;

    if (sd_command(CMD17, addr) != 0x00) {
        card_deselect();
        return -1;
    }

    uint8_t token = 0xFF;
    for (uint32_t i = 0; i < TIMEOUT_TOKEN_LOOPS && token == 0xFF; i++) {
        token = spi_byte(0xFF);
    }
    if (token != TOKEN_START_BLOCK) {
        card_deselect();
        return -2;
    }


    spi_transfer(NULL, 0, buf, 512);
    spi_transfer(NULL, 0, crc, 2);

    card_deselect();
    return 0;
}



int sd_write_block(uint32_t block, const uint8_t *buf)
{
    static uint8_t crc[2] = { 0xFF, 0xFF };
    uint32_t addr = card_is_high_capacity ? block : block * 512;

    if (sd_command(CMD24, addr) != 0x00) {
        card_deselect();
        return -1;
    }

    spi_byte(0xFF);                      /* one-byte gap before data */
    spi_byte(TOKEN_START_BLOCK);
    spi_transfer(buf, 512, NULL, 0);     /* the data, in one DMA go  */
    spi_transfer(crc, 2, NULL, 0);       /* CRC (not checked in SPI) */

    /* Data response token: lower 5 bits = 0b00101 means accepted. */
    uint8_t resp = spi_byte(0xFF);
    if ((resp & 0x1F) != 0x05) {
        card_deselect();
        return -2;
    }

    /* The card holds MISO low while it programs its flash memory. */
    if (!wait_not_busy(TIMEOUT_BUSY_LOOPS)) {
        card_deselect();
        return -3;
    }

    card_deselect();
    return 0;
}


DSTATUS disk_status(BYTE pdrv)
{
    if (pdrv != 0) {
        return STA_NOINIT;
    }
    return card_ready ? 0 : STA_NOINIT;
}

DSTATUS disk_initialize(BYTE pdrv)
{
    if (pdrv != 0) {
        return STA_NOINIT;
    }
    return (sd_init() == 0) ? 0 : STA_NOINIT;
}

DRESULT disk_read(BYTE pdrv, BYTE *buff, LBA_t sector, UINT count)
{
    if (pdrv != 0 || count == 0) {
        return RES_PARERR;
    }
    if (!card_ready) {
        return RES_NOTRDY;
    }
    for (UINT i = 0; i < count; i++) {
        if (sd_read_block((uint32_t)sector + i, buff + i * 512) != 0) {
            return RES_ERROR;
        }
    }
    return RES_OK;
}

DRESULT disk_write(BYTE pdrv, const BYTE *buff, LBA_t sector, UINT count)
{
    if (pdrv != 0 || count == 0) {
        return RES_PARERR;
    }
    if (!card_ready) {
        return RES_NOTRDY;
    }
    for (UINT i = 0; i < count; i++) {
        if (sd_write_block((uint32_t)sector + i, buff + i * 512) != 0) {
            return RES_ERROR;
        }
    }
    return RES_OK;
}

DRESULT disk_ioctl(BYTE pdrv, BYTE cmd, void *buff)
{
    (void)buff;
    if (pdrv != 0) {
        return RES_PARERR;
    }
    switch (cmd) {
    case CTRL_SYNC:
        /* sd_write_block already waits for each write to finish. */
        return RES_OK;
    default:
        /* Other commands are only needed for formatting (f_mkfs),
         * which we don't do: format the card on a PC as FAT32. */
        return RES_PARERR;
    }
}

/* FatFs stamps files with a date/time. There is no real-time clock
 * here, so use a fixed date: 2026-01-01 00:00:00.
 * (Or set FF_FS_NORTC = 1 in ffconf.h and delete this function.) */
DWORD get_fattime(void)
{
    return ((DWORD)(2026 - 1980) << 25)   /* year since 1980 */
         | ((DWORD)1 << 21)               /* month           */
         | ((DWORD)1 << 16);              /* day             */
}



static FATFS fs;          /* file-system state (~ 560 bytes of RAM)   */
static FIL   log_file;    /* the open file    (~ 560 bytes of RAM)    */
static bool  log_is_open;

int sd_log_open(const char *filename)
{
  
    if (f_mount(&fs, "", 1) != FR_OK) {
        return -1;
    }


    if (f_open(&log_file, filename, FA_WRITE | FA_OPEN_APPEND) != FR_OK) {
        f_mount(NULL, "", 0);
        return -2;
    }

    log_is_open = true;
    return 0;
}

int sd_log_write(const void *data, uint32_t len)
{
    UINT written;

    if (!log_is_open) {
        return -1;
    }

    if (f_write(&log_file, data, len, &written) != FR_OK || written != len) {
        return -2;
    }
    return 0;
}

int sd_log_sync(void)
{
    if (!log_is_open) {
        return -1;
    }

    return (f_sync(&log_file) == FR_OK) ? 0 : -2;
}

void sd_log_close(void)
{
    if (log_is_open) {
        f_close(&log_file);              /* also syncs */
        f_mount(NULL, "", 0);            /* unmount    */
        log_is_open = false;
    }
}