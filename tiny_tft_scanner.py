#!/usr/bin/env python3
import time
import numpy as np
import subprocess, json

from rtlsdr import RtlSdr

import board
from digitalio import DigitalInOut, Direction, Pull
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import st7789

# ---------- SDR SETTINGS ----------
SAMPLE_RATE = 2.4e6         # Hz
FFT_SIZE = 1024             # keep it modest for speed
START_FREQ = 462.5625e6     # Hz, starting center (FRS/GMRS-ish)
MIN_FREQ = 1e6              # clamp low end
MAX_FREQ = 1.7e9            # clamp high end (RTL range)
INITIAL_STEP = 50e3         # 50 kHz
SPECTRUM_FRACTION = 0.6     # 60% of height for bars, rest for waterfall

# Zoom presets: (label, sample_rate, default_step)
ZOOM_LEVELS = [
    ("Wide",   2.4e6, 50e3),    # ~2.4 MHz span
    ("Mid",    1.2e6, 25e3),    # ~1.2 MHz span
    ("Narrow", 3.0e5, 12.5e3),  # ~0.3 MHz span
]

zoom_index  = 0                   # start at "Wide"
sample_rate = ZOOM_LEVELS[0][1]   # will be pushed into SDR by apply_zoom()
step        = ZOOM_LEVELS[0][2]   # current tuning step (Hz)

# ---------- DISPLAY SETUP (Adafruit 1.3" TFT Bonnet) ----------
cs_pin = DigitalInOut(board.CE0)
dc_pin = DigitalInOut(board.D25)
reset_pin = DigitalInOut(board.D24)
BAUDRATE = 24000000
spi = board.SPI()

disp = st7789.ST7789(
    spi,
    height=240,
    y_offset=80,     # per Adafruit bonnet example
    rotation=180,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
)

# Backlight on pin D26
backlight = DigitalInOut(board.D26)
backlight.direction = Direction.OUTPUT
backlight.value = True

WIDTH  = disp.width
HEIGHT = disp.height

# Layout constants
HEADER_H = 40          # pixels reserved for title / text at top
# SPECTRUM_FRACTION is already defined above (e.g. 0.6)

# ---------- BUTTONS (bonnet D-pad + 2 buttons) ----------
def make_button(pin):
    b = DigitalInOut(pin)
    b.direction = Direction.INPUT
    b.pull = Pull.UP       # pressed = False
    return b

button_A = make_button(board.D5)   # top right button (not used yet)
button_B = make_button(board.D6)   # bottom right button (not used yet)

button_L = make_button(board.D27)
button_R = make_button(board.D23)
button_U = make_button(board.D17)
button_D = make_button(board.D22)
button_C = make_button(board.D4)   # center of D-pad (unused)

# ---------- IMAGE / DRAWING ----------
image = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except Exception:
    font = ImageFont.load_default()

HEADER_H = 40  # pixels reserved for text at top

# ---------- SDR SETUP ----------
sdr = RtlSdr()

# use initial zoom settings that were defined above
sdr.sample_rate = sample_rate
sdr.center_freq = START_FREQ
sdr.gain = 3    # try 10–30 dB and see what looks best

center_freq = START_FREQ

# Smoothing + peak hold
alpha = 0.3
avg_power = None                 # smoothed live spectrum
peak_power = None                # peak-hold spectrum

# Battery HUD (PiSugar)
battery_percent = 88
battery_last_update = 0.0

# Band presets: (label, center_freq_Hz, step_Hz)
PRESETS = [
    ("FRS/GMRS", 462.5625e6, 12.5e3),
    ("2m Ham",   146.520e6,  25e3),
    ("Airband",  121.500e6,  25e3),
    ("NOAA WX",  162.550e6,  12.5e3),
]

preset_index = 0
band_label = PRESETS[0][0]

# Hold + simple button edge-detect
hold = False
last_A = True   # buttons are pulled-up, so "not pressed" = True
last_B = True

def clamp_freq(f):
    return max(MIN_FREQ, min(MAX_FREQ, f))

def update_center(delta):
    global center_freq
    center_freq = clamp_freq(center_freq + delta)
    sdr.center_freq = center_freq

def update_step(up=True):
    global step
    if up:
        step = min(step * 2, 5e6)   # max 5 MHz
    else:
        step = max(step / 2, 1e3)   # min 1 kHz

def apply_zoom():
    """Update SDR sample_rate and step when zoom level changes."""
    global sample_rate, step, zoom_index, sdr
    label, sr, default_step = ZOOM_LEVELS[zoom_index]
    sample_rate = sr
    step        = default_step
    sdr.sample_rate = sample_rate

def cycle_zoom():
    """Cycle through zoom levels (Wide -> Mid -> Narrow -> Wide...)."""
    global zoom_index
    zoom_index = (zoom_index + 1) % len(ZOOM_LEVELS)
    apply_zoom()

def get_battery_percent_raw():
    """
    Try to read PiSugar battery percentage via its HTTP API.
    Requires pisugar-server running (default: http://127.0.0.1:8421/state).
    Returns int 0-100 or None if unavailable.
    """
    try:
        out = subprocess.check_output(
            ["curl", "-s", "http://127.0.0.1:8421/state"],
            timeout=0.3,
        )
        data = json.loads(out)
        b = data.get("battery") or {}
        p = b.get("percentage")
        if p is None:
            return None
        return int(p)
    except Exception:
        return None


def update_battery():
    """Poll PiSugar every few seconds and cache the %."""
    global battery_percent, battery_last_update
    now = time.time()
    if now - battery_last_update < 5:
        return
    battery_last_update = now
    bp = get_battery_percent_raw()
    if bp is not None:
        battery_percent = bp


def level_to_color(v):
    """
    Map 0..1 power level to a simple waterfall color.
    Dark -> green -> yellow -> almost white.
    """
    v = max(0.0, min(1.0, float(v)))
    r = int(255 * v)
    g = int(255 * min(1.0, v * 1.3))
    b = int(255 * max(0.0, (v - 0.4) * 1.8))
    return (r, g, b)


def update_waterfall(column_vals, wf_top, wf_h):
    """
    Scroll the waterfall region up by 1 and draw a new row at the bottom
    using the per-column amplitudes in column_vals (0..1).
    """
    global image
    if wf_h <= 0:
        return

    # Scroll waterfall up by 1 pixel
    src_box = (0, wf_top + 1, WIDTH, wf_top + wf_h)
    region = image.crop(src_box)
    image.paste(region, (0, wf_top))

    # Draw newest row at the bottom
    y = wf_top + wf_h - 1
    for x, v in enumerate(column_vals):
        image.putpixel((x, y), level_to_color(v))

def draw_spectrum(power_db):
    global peak_power, center_freq, step, hold, band_label, battery_percent

    # Estimate noise floor as ~60th percentile
    noise_floor = float(np.percentile(power_db, 60))

    # Define visible range around the floor
    min_level = noise_floor - 5    # a bit below floor
    max_level = noise_floor + 25   # strong signals

    # Clip and normalize to 0..1 for live spectrum
    power_clip = np.clip(power_db, min_level, max_level)
    norm = (power_clip - min_level) / (max_level - min_level)

    # Peak-hold normalization if we have it
    peak_norm = None
    if peak_power is not None:
        peak_clip = np.clip(peak_power, min_level, max_level)
        peak_norm = (peak_clip - min_level) / (max_level - min_level)

    # Layout: spectrum on top, waterfall underneath
    total_h = HEIGHT - HEADER_H
    spec_h = int(total_h * SPECTRUM_FRACTION)
    wf_h = total_h - spec_h

    spec_top = HEADER_H
    wf_top = spec_top + spec_h

    # Clear ONLY header + spectrum region (leave waterfall)
    draw.rectangle((0, 0, WIDTH, spec_top + spec_h), outline=0, fill=(0, 0, 0))

    # Header text
    cf_mhz = center_freq / 1e6
    step_khz = step / 1e3
    status = "HOLD" if hold else band_label
    draw.text((2, 2),  f"{status}  {cf_mhz:8.4f} MHz", font=font, fill=(0, 255, 0))
    draw.text((2, 12), f"Step: {step_khz:5.1f} kHz",   font=font, fill=(0, 200, 255))
    bw_mhz = sample_rate / 1e6
    draw.text((2, 26), f"BW: {bw_mhz:4.2f} MHz", font=font, fill=(0, 200, 255))


    # Battery HUD in top-right
    if battery_percent is not None:
        btxt = f"{battery_percent:3d}%"

        # Pillow 10+ removed textsize(), use textbbox() instead
        bbox = draw.textbbox((0, 0), btxt, font=font)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]

        draw.text((WIDTH - bw - 2, 2), btxt, font=font, fill=(0, 255, 255))


    # For each screen column, take the MAX of its bin range
    bins_per_pixel = FFT_SIZE / WIDTH
    column_vals = [0.0] * WIDTH

    for x in range(WIDTH):
        start = int(x * bins_per_pixel)
        end = int((x + 1) * bins_per_pixel)
        if end <= start:
            end = start + 1

        v_live = float(np.max(norm[start:end]))  # live
        column_vals[x] = v_live

        # Live bar
        bar_h = int(v_live * spec_h)
        y0 = spec_top + (spec_h - bar_h)
        y1 = spec_top + spec_h
        draw.line((x, y0, x, y1), fill=(0, 255, 0))

        # Peak-hold marker (tiny yellow dot), only for *strong* peaks
        if peak_norm is not None:
            pval = float(np.max(peak_norm[start:end]))
            # norm is 0..1 over ~30 dB span; ~0.4 ≈ ~7–8 dB above floor
            if pval > 0.4:
                py = spec_top + (spec_h - int(pval * spec_h))
                draw.point((x, py), fill=(255, 255, 0))

    # Update waterfall with this frame's column values
    update_waterfall(column_vals, wf_top, wf_h)

    # Push to display
    disp.image(image)

def main():
    global center_freq, step, avg_power, peak_power, hold, preset_index, band_label
    global zoom_index, sample_rate, last_A, last_B


    # Make sure SDR sample rate matches current zoom on startup
    apply_zoom()

    try:
        while True:
            # Update battery HUD (PiSugar)
            update_battery()

            # ----- BUTTON HANDLING -----
            # D-pad for navigation (same as before)
            if not button_L.value:
                update_center(-step)
            if not button_R.value:
                update_center(+step)
            if not button_U.value:
                update_step(up=True)
            if not button_D.value:
                update_step(up=False)

            # Edge-detect on A/B buttons (top/bottom right)
            newA = button_A.value       # pulled-up, so False = pressed
            newB = button_B.value

            # Both right-side buttons pressed together -> cycle zoom
            if last_A and last_B and (not newA) and (not newB):
                cycle_zoom()

            # Only A pressed -> cycle band presets
            elif last_A and not newA:
                preset_index = (preset_index + 1) % len(PRESETS)
                label, cf, st = PRESETS[preset_index]
                band_label = label
                center_freq = clamp_freq(cf)
                step = st
                sdr.center_freq = center_freq
                # reset smoothing when jumping bands
                avg_power = None
                peak_power = None

            # Only B pressed -> HOLD toggle
            elif last_B and not newB:
                hold = not hold

            # remember for next loop
            last_A = newA
            last_B = newB

            # ----- SDR READ + DRAW -----
            if not hold:
                # Get samples
                samples = sdr.read_samples(FFT_SIZE * 2)

                # Hann window for cleaner FFT
                window = np.hanning(FFT_SIZE)
                samples_w = samples[:FFT_SIZE] * window

                # FFT and power in dB
                fft = np.fft.fftshift(np.fft.fft(samples_w, n=FFT_SIZE))
                power_db = 10 * np.log10(np.abs(fft) ** 2 + 1e-12)

                # Smoothing
                if avg_power is None:
                    avg_power = power_db
                else:
                    avg_power = alpha * power_db + (1.0 - alpha) * avg_power

                # Peak hold
                if peak_power is None:
                    peak_power = avg_power.copy()
                else:
                    peak_power = np.maximum(peak_power, avg_power)

                # Draw with waterfall + battery HUD
                if avg_power is not None:
                    draw_spectrum(avg_power)

            else:
                # HOLD: just redraw last frame with updated labels/battery
                if avg_power is not None:
                    draw_spectrum(avg_power)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        sdr.close()
        draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=(0, 0, 0))
        disp.image(image)


if __name__ == "__main__":
    main()


