# Tiny Specan

Tiny Specan is a pocket-sized spectrum analyzer built around a Raspberry Pi Zero 2W, an Adafruit 1.3" Color TFT Bonnet, and an RTL-SDR dongle. It gives you a live RF "heads-up display" in your hand: spectrum, simple waterfall, peak-hold, and quick controls for step/BW.

> **Use case:** throw it in a bag, power it from a USB bank or PiSugar, and sweep common bands for activity without needing a laptop.

---

## Hardware

- **Compute:** Raspberry Pi Zero 2W  
- **Display:** Adafruit 1.3" Color TFT Bonnet for Raspberry Pi  
- **SDR:** RTL-SDR dongle (e.g., Nooelec NESDR)  
- **Power (optional):** PiSugar or USB power bank  
- **Antenna:** Any antenna appropriate for the bands you’re watching

Adjust this list to match your exact build (case, battery, etc.).

---

## Features

- Real-time FFT plot on a 1.3" TFT screen
- Rolling / hold trace to show recent peaks
- Center frequency, span / step, and bandwidth controls
- On-screen indicators for:
  - Center frequency
  - Step size
  - Bandwidth (BW)
  - Hold / averaging state
- Optimized for headless / field use – boots straight into the scanner script

---

## Software / Dependencies

On the Pi (Debian / Raspberry Pi OS 64-Bit Lite):

- Python 3
- `numpy`
- `matplotlib` (if you use it for any off-screen plotting or debug)
- RTL-SDR tools / libraries (e.g. `rtl-sdr`, `librtlsdr`)
- `pillow` or other imaging libs if your code uses them
- Adafruit libraries for the 1.3" TFT Bonnet (SPI display + buttons)

Example:

```bash
sudo apt update
sudo apt install -y python3 python3-pip rtl-sdr
pip3 install numpy
# Add any others your script actually uses:
# pip3 install pillow
