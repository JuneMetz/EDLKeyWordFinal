/*
 *
 *     P0.14 (F4) -> SD CLK  (SCK)
 *     P0.16 (G4) -> SD CMD  (MOSI, data from nRF to card)
 *     P0.18 (G3) <- SD DAT0 (MISO, data from card to nRF)
 *     P0.20 (F2) -> SD DAT3 (CS, chip select, active low)
 *     plus 3.3 V and GND to the card
 *
 * Usage:
 *     sd_log_open("KWS_LOG.CSV");        // once, at startup
 *     sd_log_write("123,4,567\n", 10);   // as often as needed
 *     sd_log_sync();                     // every few seconds
 *     sd_log_close();                    // before power-off / removal
 *
 * Needs the FatFs library (ff.c, ff.h, ffconf.h, diskio.h) from
 * http://elm-chan.org/fsw/ff/ . This file provides the disk_xxx()
 * functions FatFs calls, so do NOT also compile FatFs's template
 * diskio.c.
 */
#ifndef SD_CARD_H
#define SD_CARD_H

#include <stdint.h>

#define SD_PIN_SCK    14
#define SD_PIN_MOSI   16
#define SD_PIN_MISO   18
#define SD_PIN_CS     20


/* Set up the SPI pins and the SPIM0 peripheral at a given speed.
 * freq is one of SPIM_FREQUENCY_FREQUENCY_K125 ... _M8 from nrf.h. */
void sd_spi_config(uint32_t freq);

/* Wake the card and put it in SPI mode. Returns 0 on success. */
int  sd_init(void);

/* Read / write one 512-byte block. 'block' is the block number
 * (block 0 = first 512 bytes of the card). Returns 0 on success. */
int  sd_read_block(uint32_t block, uint8_t *buf);
int  sd_write_block(uint32_t block, const uint8_t *buf);


/* Mount the card and open (or create) a file for appending.
 * Use an 8.3 file name, e.g. "KWS_LOG.CSV". Returns 0 on success. */
int  sd_log_open(const char *filename);

/* Append bytes to the file. Returns 0 on success. */
int  sd_log_write(const void *data, uint32_t len);

/* Force everything written so far onto the card. Returns 0 on success.
 * Until this runs, data may only be in RAM and lost on power-off. */
int  sd_log_sync(void);

/* Close the file and unmount the card. */
void sd_log_close(void);

#endif /* SD_CARD_H */
