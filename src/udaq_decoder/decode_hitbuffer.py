#!/usr/bin/python
import struct
from typing import Callable, Iterator, Any
from dataclasses import dataclass

__all__ = ['Hit', 'decode_hitbuffer', 'fix_hitbuffer_data']

OBJECT_CODE_PPS_SECOND  = 0xe0
OBJECT_MASK_PPS_SECOND  = 0xfc
OBJECT_CODE_PPS_YEAR    = 0xe4
OBJECT_CODE_TRIG_CONFIG = 0xe5
OBJECT_CODE_DATA_FORMAT = 0xe6
OBJECT_CODE_PAGE_END    = 0xe7
OBJECT_CODE_GENERIC     = 0xf0
OBJECT_MASK_GENERIC     = 0xf0

STATUS_CPUTRIG_ACTIVE = (1<<5)
DATA_FORMAT_TIMESTAMP = 1
DETAIL_TIMESTAMP_FINE = (1<<0)
DATA_FORMAT_TIMESTAMP_TOT_ADCS = 2
DATA_FORMAT_TIMESTAMP_TOT_ALL_CCRS = 3

class RawFrame:
    pass

@dataclass
class RawFramePpsYear(RawFrame):
    year: int

def _decode_pps_year(header: int, words: Iterator[int]):
    year = header & 0xFFFF
    return RawFramePpsYear(year)

@dataclass
class RawFramePpsSecond(RawFrame):
    second: int

def _decode_pps_second(header: int, words: Iterator[int]):
    second = header & 0x03ffffff
    return RawFramePpsSecond(second)

@dataclass
class RawFrameTriggerConf(RawFrame):
    rc_status: int # cpu trigger?
    mode: int
    offset: int

def _decode_trigger_config(header: int, words: Iterator[int]):
    mode = (header >> 16) & 0xff
    rc_status = header & 0xffff
    offset = next(words)
    return RawFrameTriggerConf(rc_status, mode, offset)

@dataclass
class RawFrameDataFormat(RawFrame):
    subtype: int
    detail_bits: int

def _decode_data_format(header: int, words: Iterator[int]):
    subtype = (header >> 16) & 0xff
    detail_bits = header & 0xffff
    return RawFrameDataFormat(subtype, detail_bits)

@dataclass
class RawFrameHit(RawFrame):
    offset: int
    tot: int
    adc_data: dict[int, int]

def _decode_hit_tot_adcs(header: int, words: Iterator[int]):
    offset = header
    # decode second word
    multi = next(words)
    adc_count = (multi >> 28) & 0xf
    tot = (multi >> 16) & 0xfff
    # decode ADC data
    adc_data : dict(int, int) = {}
    adc_odd = True
    adc_word = (multi & 0xffff) << 16
    for _ in range(adc_count):
        if adc_odd:
            adc_word >>= 16
        else:
            adc_word = next(words)
        adc_odd = not adc_odd
        adc_index = (adc_word >> 12) & 0xf
        adc_value = adc_word & 0xfff
        adc_data[adc_index] = adc_value
    return RawFrameHit(offset, tot, adc_data)


decoders : dict[int, Callable[[int, Iterator[int]], Any]] = {
    OBJECT_CODE_PPS_YEAR: _decode_pps_year,
    OBJECT_CODE_PPS_SECOND: _decode_pps_second,
    OBJECT_CODE_TRIG_CONFIG: _decode_trigger_config,
    OBJECT_CODE_DATA_FORMAT: _decode_data_format,
}

def resolve_time_10th_ns(second: int, offset: int) -> int:
    return (second * int(1e10)) + ((offset * 125) // 36)

@dataclass
class Hit:
    year: int
    ns_10th: int
    adc_data: dict[int, int]
    tot: int # time over threshold
    cpu_trigger: bool

def decode_hitbuffer(data_raw: bytes, ignore_errors: bool = False) -> list[Hit]:
    # unpack tuples returned by iter_unpack
    words = (value for value, in struct.iter_unpack('<I', data_raw))
    # decode frames
    frames = []
    try:
        for header in words:
            frame_type = header >> 24
            if frame_type in decoders:
                frames.append(decoders[frame_type](header, words))
            elif frame_type < 0xdfffffff:
                frames.append(_decode_hit_tot_adcs(header, words))
    except StopIteration:
        if not ignore_errors:
            raise ValueError('Incomplete frame at end of data')
    first_hit_adc_keys = next(frame.adc_data.keys() for frame in frames if isinstance(frame, RawFrameHit))
    # combine data from frames
    year = -1
    second = -1
    cpu_trigger = False
    hits : list[Hit] = []
    for frame in frames:
        if isinstance(frame, RawFramePpsYear):
            year = frame.year
        elif isinstance(frame, RawFramePpsSecond):
            second = frame.second
        elif isinstance(frame, RawFrameTriggerConf):
            cpu_trigger = (frame.rc_status & STATUS_CPUTRIG_ACTIVE) != 0
        elif isinstance(frame, RawFrameHit):
            if frame.adc_data.keys() != first_hit_adc_keys:
                if not ignore_errors:
                    raise ValueError(f'hit contains ADC keys different from first hit: {frame.adc_data.keys()}')
                continue
            hits.append(Hit(
                year,
                resolve_time_10th_ns(second, frame.offset),
                frame.adc_data,
                frame.tot,
                cpu_trigger,
            ))
    return hits

# fix hitbuffer data missing two bytes every 1024 bytes (firmware or readout error in uDAQ)
def fix_hitbuffer_data(data_raw: bytes):
    data_raw = bytearray(data_raw)
    for i in range(1024, len(data_raw), 1024):
        data_raw.insert(i, 0)
        data_raw.insert(i, 0)
    while len(data_raw) % 4 != 0:
        data_raw.append(0)
    return bytes(data_raw)
