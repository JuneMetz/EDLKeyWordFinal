/*
 * adc.h - 12-bit differential audio capture on the nRF52805 SAADC
 *
 * Input : P0.04 (AIN2) = positive, P0.05 (AIN3) = negative
 * Rate  : 16 kHz, 12-bit, delivered in blocks of ADC_BUF_LEN samples
 *

 */
#ifndef ADC_H
#define ADC_H

#include <stdint.h>

#define ADC_SAMPLE_RATE_HZ  16000  
#define ADC_BUF_LEN         320    


void adc_config(void);

/* Start / stop continuous sampling. */
void adc_start(void);
void adc_stop(void);

/* Returns a pointer to a full buffer of ADC_BUF_LEN samples,
 * or NULL if no new buffer is ready yet. */
int16_t *adc_get_full_buffer(void);

/* Tell the driver sampling finished with the buffer from
 * adc_get_full_buffer(). Must be called within ~20 ms. */
void adc_release_buffer(void);

/* Number of times a full buffer was not released in time
 * (i.e. processing was too slow and audio was lost). */
uint32_t adc_get_overrun_count(void);

#endif 