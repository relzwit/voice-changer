#ifndef VOICE_CHANGER_AUDIOIO_H
#define VOICE_CHANGER_AUDIOIO_H

#include <miniaudio.h>
#include "Utils.h"

// Shared Ring Buffers
extern AudioRingBuffer g_rb_input;
extern AudioRingBuffer g_rb_output;

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

#endif //VOICE_CHANGER_AUDIOIO_H