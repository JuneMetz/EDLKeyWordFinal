/*
 * adc.c - 12-bit differential audio capture on the nRF52805 SAADC
 *
 * Register-level code based on the nRF52805 Product Specification v1.2.
 * Section numbers in the comments ("PS 6.14.x") refer to that document.
 *
 * ---------------------------------------------------------------------
 * HOW IT WORKS (big picture)
 * ---------------------------------------------------------------------
 *
 *   TIMER2 ---(every 62.5 us)---> PPI ch0 ---> SAADC SAMPLE task
 *                                                   |
 *                                       one 12-bit sample converted
 *                                                   |
 *                         EasyDMA writes it into the current RAM buffer
 *                                                   |
 *            buffer full (ADC_BUF_LEN samples) -> SAADC END event
 *                                                   |
 *                 PPI ch1 ---> SAADC START task (immediately begins the
 *                 next buffer, in hardware, so no samples are missed)
 *                                                   |
 *                 SAADC interrupt -> marks the full buffer as ready
 *
 * We use TWO buffers ("ping-pong"): while the SAADC fills one, your
 * code reads the other. Each buffer lasts 20 ms, so your processing
 * of one buffer must finish in under 20 ms.
 *
 * Why a TIMER + PPI and not the CPU? The CPU never has to touch
 * individual samples: the hardware takes 16000 samples per second
 * on its own, and the CPU is only interrupted 50 times per second
 * (once per full buffer).
 * ---------------------------------------------------------------------
 */

#include <stdbool.h>
#include <stddef.h>
#include "nrf.h"      /* register definitions (Nordic MDK) */
#include "adc.h"

/* The 16 MHz timer counts this many ticks between samples:
 * 16,000,000 / 16,000 = 1000 ticks = 62.5 us per sample. */
#define TIMER_TICKS_PER_SAMPLE  (16000000UL / ADC_SAMPLE_RATE_HZ)

/* PPI channels used. Any free channels 0..19 work; if you add a
 * SoftDevice (Bluetooth stack) later, it reserves some of the higher
 * numbered ones, so low numbers are a safe choice. */
#define PPI_CH_SAMPLE   0   /* TIMER2 COMPARE[0] -> SAADC SAMPLE */
#define PPI_CH_RESTART  1   /* SAADC END         -> SAADC START  */

/* ---------------------------------------------------------------------
 * Buffers and state shared between the interrupt and your code
 * ---------------------------------------------------------------------
 * 'volatile' tells the compiler these can change at any moment
 * (inside the interrupt), so it must always re-read them. */

/* The two ping-pong buffers. Must be in RAM because EasyDMA can only
 * write to RAM (PS 6.14.5). Each sample is 16-bit signed. */
static int16_t adc_buffers[2][ADC_BUF_LEN];

/* Which buffer (0 or 1) the SAADC is writing right now. */
static volatile uint8_t filling_index;

/* Which buffer is loaded in RESULT.PTR for the *next* START. */
static volatile uint8_t next_index;

/* Points to the buffer that is full and waiting for your code.
 * NULL means nothing new is ready. */
static int16_t * volatile ready_buffer;

/* Counts how often a buffer filled up before you released the
 * previous one (= audio lost because processing was too slow). */
static volatile uint32_t overrun_count;


/* =====================================================================
 * adc_config()
 * Sets up every register needed. Call once at startup, before
 * adc_start().
 * ===================================================================== */
void adc_config(void)
{
    /* -----------------------------------------------------------------
     * 1. Start the 32 MHz crystal (HFXO).
     *    TIMER2 counts from the high-frequency clock. The internal RC
     *    oscillator is less accurate, which would make the 16 kHz
     *    sample rate drift. The crystal keeps it exact.
     * ----------------------------------------------------------------- */
    NRF_CLOCK->EVENTS_HFCLKSTARTED = 0;
    NRF_CLOCK->TASKS_HFCLKSTART = 1;
    while (NRF_CLOCK->EVENTS_HFCLKSTARTED == 0) {
        /* wait for the crystal to stabilise (well under 1 ms) */
    }

    /* -----------------------------------------------------------------
     * 2. SAADC must be disabled while we change its configuration.
     * ----------------------------------------------------------------- */
    NRF_SAADC->ENABLE = SAADC_ENABLE_ENABLE_Disabled;

    /* Disconnect all 8 channels first so only channel 0 is active.
     * (Only one channel may be enabled for continuous sampling.) */
    for (int ch = 0; ch < 8; ch++) {
        NRF_SAADC->CH[ch].PSELP = SAADC_CH_PSELP_PSELP_NC;
        NRF_SAADC->CH[ch].PSELN = SAADC_CH_PSELN_PSELN_NC;
    }

    /* -----------------------------------------------------------------
     * 3. Pick the input pins for channel 0 (PS 6.14.10.18 / 6.14.10.19)
     *    PSELP = positive input, PSELN = negative input.
     *    From the pin table (PS Table 104):
     *       AIN2 = P0.04 (ball F5)  -> positive
     *       AIN3 = P0.05 (ball F6)  -> negative
     * ----------------------------------------------------------------- */
    NRF_SAADC->CH[0].PSELP = SAADC_CH_PSELP_PSELP_AnalogInput2;
    NRF_SAADC->CH[0].PSELN = SAADC_CH_PSELN_PSELN_AnalogInput3;

    /* -----------------------------------------------------------------
     * 4. Channel 0 settings (CH[0].CONFIG, PS 6.14.10.20)
     *
     *  RESP / RESN = Bypass
     *      No internal pull-up/pull-down resistors on the inputs.
     *
     *  GAIN = 1/4 and REFSEL = VDD/4
     *      Input range = +/- (reference / gain) = +/- (VDD/4)/(1/4)
     *                  = +/- VDD  (PS 6.14.7)
     *      Raise GAIN (e.g. 1/2 -> +/- VDD/2) if your mic signal is
     *      small and you want more resolution. Each pin must always
     *      stay between 0 V and VDD.
     *
     *  TACQ = 10 us
     *      Time the ADC spends charging its sample capacitor. 10 us
     *      works for a source resistance up to 100 kOhm. 10 us + ~2 us
     *      conversion = 12 us, well under the 62.5 us between samples.
     *
     *  MODE = Diff
     *      Measures V(P0.04) - V(P0.05). Noise that appears on both
     *      wires equally cancels out.
     *
     *  BURST = Disabled
     *      One conversion per SAMPLE trigger (no averaging).
     * ----------------------------------------------------------------- */
    NRF_SAADC->CH[0].CONFIG =
          (SAADC_CH_CONFIG_RESP_Bypass    << SAADC_CH_CONFIG_RESP_Pos)
        | (SAADC_CH_CONFIG_RESN_Bypass    << SAADC_CH_CONFIG_RESN_Pos)
        | (SAADC_CH_CONFIG_GAIN_Gain1_4   << SAADC_CH_CONFIG_GAIN_Pos)
        | (SAADC_CH_CONFIG_REFSEL_VDD1_4  << SAADC_CH_CONFIG_REFSEL_Pos)
        | (SAADC_CH_CONFIG_TACQ_10us      << SAADC_CH_CONFIG_TACQ_Pos)
        | (SAADC_CH_CONFIG_MODE_Diff      << SAADC_CH_CONFIG_MODE_Pos)
        | (SAADC_CH_CONFIG_BURST_Disabled << SAADC_CH_CONFIG_BURST_Pos);

    /* -----------------------------------------------------------------
     * 5. Resolution and oversampling
     *    12-bit (PS 6.14.10.22). In differential mode the result is
     *    signed: -2048 .. +2047 (PS 6.14.2, m = 1 for Diff).
     *    No oversampling (PS 6.14.10.23).
     * ----------------------------------------------------------------- */
    NRF_SAADC->RESOLUTION = SAADC_RESOLUTION_VAL_12bit;
    NRF_SAADC->OVERSAMPLE = SAADC_OVERSAMPLE_OVERSAMPLE_Bypass;

    /* -----------------------------------------------------------------
     * 6. Sample-rate control (PS 6.14.10.24)
     *    MODE = Task: a sample is taken each time the SAMPLE task is
     *    triggered. TIMER2 will trigger it through the PPI.
     * ----------------------------------------------------------------- */
    NRF_SAADC->SAMPLERATE =
        (SAADC_SAMPLERATE_MODE_Task << SAADC_SAMPLERATE_MODE_Pos);

    /* -----------------------------------------------------------------
     * 7. Enable the SAADC and run offset calibration once.
     *    Calibration measures the ADC's internal offset error and
     *    removes it from later results.
     * ----------------------------------------------------------------- */
    NRF_SAADC->ENABLE = SAADC_ENABLE_ENABLE_Enabled;

    NRF_SAADC->EVENTS_CALIBRATEDONE = 0;
    NRF_SAADC->TASKS_CALIBRATEOFFSET = 1;
    while (NRF_SAADC->EVENTS_CALIBRATEDONE == 0) {
        /* wait */
    }
    NRF_SAADC->EVENTS_CALIBRATEDONE = 0;

    /* Make sure the SAADC is fully stopped before the first START. */
    NRF_SAADC->EVENTS_STOPPED = 0;
    NRF_SAADC->TASKS_STOP = 1;
    while (NRF_SAADC->EVENTS_STOPPED == 0) {
        /* wait */
    }
    NRF_SAADC->EVENTS_STOPPED = 0;

    /* -----------------------------------------------------------------
     * 8. Tell EasyDMA where to put the samples (PS 6.14.10.25 / .26)
     *    RESULT.PTR    = address of the buffer in RAM
     *    RESULT.MAXCNT = how many samples fit (END fires when full)
     * ----------------------------------------------------------------- */
    NRF_SAADC->RESULT.PTR    = (uint32_t)adc_buffers[0];
    NRF_SAADC->RESULT.MAXCNT = ADC_BUF_LEN;
    next_index    = 0;
    filling_index = 0;
    ready_buffer  = NULL;
    overrun_count = 0;

    /* -----------------------------------------------------------------
     * 9. Interrupts: we want to know about two events
     *    STARTED - the SAADC has latched RESULT.PTR, so we can load
     *              the address of the *next* buffer (PS 6.14.5)
     *    END     - the current buffer is full
     * ----------------------------------------------------------------- */
    NRF_SAADC->EVENTS_STARTED = 0;
    NRF_SAADC->EVENTS_END     = 0;
    NRF_SAADC->INTENCLR = 0xFFFFFFFF;                    /* all off */
    NRF_SAADC->INTENSET = SAADC_INTENSET_STARTED_Msk     /* then on */
                        | SAADC_INTENSET_END_Msk;

    NVIC_SetPriority(SAADC_IRQn, 1);     /* high priority (0 = highest) */
    NVIC_ClearPendingIRQ(SAADC_IRQn);
    NVIC_EnableIRQ(SAADC_IRQn);

    /* -----------------------------------------------------------------
     * 10. TIMER2: the 16 kHz "metronome"
     *     PRESCALER = 0  -> counts at 16 MHz / 2^0 = 16 MHz
     *     CC[0]     = 1000 -> COMPARE[0] event every 1000 ticks
     *                         = every 62.5 us = 16 kHz
     *     SHORTS    = COMPARE0 -> CLEAR, so the counter resets to 0
     *                 each time and repeats forever.
     * ----------------------------------------------------------------- */
    NRF_TIMER2->TASKS_STOP  = 1;
    NRF_TIMER2->TASKS_CLEAR = 1;
    NRF_TIMER2->MODE      = TIMER_MODE_MODE_Timer;
    NRF_TIMER2->BITMODE   = TIMER_BITMODE_BITMODE_16Bit;
    NRF_TIMER2->PRESCALER = 0;
    NRF_TIMER2->CC[0]     = TIMER_TICKS_PER_SAMPLE;
    NRF_TIMER2->SHORTS    = TIMER_SHORTS_COMPARE0_CLEAR_Msk;

    /* -----------------------------------------------------------------
     * 11. PPI (Programmable Peripheral Interconnect)
     *     A PPI channel connects an EVENT of one peripheral to a TASK
     *     of another, entirely in hardware, with no CPU involvement.
     *     EEP = Event End Point (address of the event register)
     *     TEP = Task  End Point (address of the task register)
     * ----------------------------------------------------------------- */

    /* ch0: every TIMER2 tick -> take one ADC sample */
    NRF_PPI->CH[PPI_CH_SAMPLE].EEP = (uint32_t)&NRF_TIMER2->EVENTS_COMPARE[0];
    NRF_PPI->CH[PPI_CH_SAMPLE].TEP = (uint32_t)&NRF_SAADC->TASKS_SAMPLE;

    /* ch1: buffer full -> immediately start filling the next buffer.
     * The SAADC has no built-in shortcut for this, so PPI does it. */
    NRF_PPI->CH[PPI_CH_RESTART].EEP = (uint32_t)&NRF_SAADC->EVENTS_END;
    NRF_PPI->CH[PPI_CH_RESTART].TEP = (uint32_t)&NRF_SAADC->TASKS_START;
}


/* =====================================================================
 * adc_start()
 * Begins continuous sampling.
 * ===================================================================== */
void adc_start(void)
{
    /* Turn on both PPI connections. */
    NRF_PPI->CHENSET = (1UL << PPI_CH_SAMPLE) | (1UL << PPI_CH_RESTART);

    /* Prepare the SAADC with the first buffer... */
    NRF_SAADC->TASKS_START = 1;

    /* ...then start the metronome. From here on, everything runs
     * by itself; the CPU only sees one interrupt per full buffer. */
    NRF_TIMER2->TASKS_START = 1;
}


/* =====================================================================
 * adc_stop()
 * Stops sampling cleanly.
 * ===================================================================== */
void adc_stop(void)
{
    NRF_TIMER2->TASKS_STOP = 1;                      /* no more SAMPLEs  */
    NRF_PPI->CHENCLR = (1UL << PPI_CH_SAMPLE)        /* unlink hardware  */
                     | (1UL << PPI_CH_RESTART);

    NRF_SAADC->EVENTS_STOPPED = 0;
    NRF_SAADC->TASKS_STOP = 1;
    while (NRF_SAADC->EVENTS_STOPPED == 0) {
        /* wait */
    }
    NRF_SAADC->EVENTS_STOPPED = 0;
}


/* =====================================================================
 * SAADC_IRQHandler()
 * The interrupt that runs when the DMA buffer is full (END) or a new
 * buffer has started (STARTED).
 *
 * The name SAADC_IRQHandler is the standard vector name from Nordic's
 * startup file, so it is called automatically; you never call it.
 * (If you use Zephyr instead of bare metal, rename it and register it
 *  with IRQ_CONNECT(SAADC_IRQn, 1, your_isr, NULL, 0).)
 *
 * Keep interrupts short: we only swap pointers and set a flag here.
 * The heavy work happens in your main loop.
 * ===================================================================== */
void SAADC_IRQHandler(void)
{
    /* ---------------- Buffer full ----------------
     * Handle END before STARTED: if both are pending at once, END
     * belongs to the buffer that was filling *before* the new START. */
    if (NRF_SAADC->EVENTS_END) {
        NRF_SAADC->EVENTS_END = 0;
        (void)NRF_SAADC->EVENTS_END;   /* read back so the clear takes
                                          effect before we return     */

        /* If the previous buffer was never released, your code is
         * running too slowly and that audio has been lost. */
        if (ready_buffer != NULL) {
            overrun_count++;
        }

        /* Hand the full buffer to the main loop. */
        ready_buffer = adc_buffers[filling_index];

        /* Note: PPI ch1 has already triggered START in hardware, so
         * the SAADC is already filling the other buffer. */
    }

    /* ---------------- New buffer started ----------------
     * RESULT.PTR is "double-buffered": once STARTED fires, the SAADC
     * has copied the address it is using, and we may load the address
     * for the NEXT buffer without disturbing the current one. */
    if (NRF_SAADC->EVENTS_STARTED) {
        NRF_SAADC->EVENTS_STARTED = 0;
        (void)NRF_SAADC->EVENTS_STARTED;

        filling_index = next_index;         /* the one just started   */
        next_index ^= 1;                    /* 0 -> 1, 1 -> 0         */
        NRF_SAADC->RESULT.PTR = (uint32_t)adc_buffers[next_index];
    }
}


/* =====================================================================
 * Collection functions - call these from your main loop
 * ===================================================================== */

/* Returns the full buffer, or NULL if nothing new yet.
 * Samples are signed 12-bit values (-2048 .. +2047) stored in int16_t:
 *     -2048 = P0.04 is VDD below P0.05
 *         0 = both pins at the same voltage
 *     +2047 = P0.04 is VDD above P0.05                                */
int16_t *adc_get_full_buffer(void)
{
    return ready_buffer;
}

/* Call when you are done reading the buffer, so the next one can be
 * handed over. You have ~20 ms (one buffer time) to do so. */
void adc_release_buffer(void)
{
    ready_buffer = NULL;
}

uint32_t adc_get_overrun_count(void)
{
    return overrun_count;
}