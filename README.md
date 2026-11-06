# Arc reactor

A chest-worn voice client for JARVIS MK3. It listens for **"hey jarvis"**
(or a press of the earphones' play/pause button), records the question
until you pause, sends it to the JARVIS Core over
the client WebSocket, and plays the spoken reply. Nothing leaves the
device until the wake word fires.

Prototype board: Raspberry Pi 3 Model B+. Target: Pi Zero 2 W (512 MB),
so the client stays around 100 MB.

## How it fits together

```
 mic (parec) ──80 ms chunks──> wake word (3 ONNX models) ──score──┐
                      └──────> background level / preroll         │
                                                                  v
 link (WebSocket) <──audio_chunk/audio_end── app (state machine) ──> lights
        │                                       ^        │
        └──transcript/audio/done/notification───┘        └──> player (pacat)
 router (pactl, every 4 s): which mic, which speaker, earphone profile
```

| file | job |
|---|---|
| `reactor/app.py` | the state machine: offline, idle, listening, thinking, speaking |
| `reactor/wake.py` | openWakeWord's pipeline run directly on onnxruntime |
| `reactor/listen.py` | where a question starts and ends; WAV and frames |
| `reactor/mic.py` | `parec` on the chosen mic; restarts when it changes |
| `reactor/player.py` | one stream per reply, sentences in order, lead-in silence |
| `reactor/link.py` | the Core connection, hello, reconnect forever |
| `reactor/buttons.py` | the earphones' buttons, read from BlueZ's "(AVRCP)" input device |
| `reactor/routes.py` | device choice (bluetooth > usb > onboard) and earphone profile |
| `reactor/sounds.py` | the chimes |
| `reactor/lights.py` | LED interface (logs only until the rings arrive) |
| `reactor/config.py` | defaults and `~/.config/reactor/config.toml` |

## Commands

```
python -m reactor                 # run (the service does this)
python -m reactor check           # token, models, audio devices, connection
python -m reactor scores          # live wake word scores, for tuning the threshold
python -m reactor buttons         # show which earphone buttons arrive
python -m reactor ask q.wav       # send one recorded question and play the reply
python -m reactor --mic-file f.wav   # run, hearing a file instead of the mic
```

## The earphone buttons

| button | does |
|---|---|
| play/pause | start a question (cuts JARVIS off if he is talking); press again to send at once |
| previous | drop the question being recorded, or stop JARVIS mid-sentence |
| next | wake word on / off (off: the button is the only trigger, and the detector stops using CPU) |

Needs, once: the `uinput` module loaded (`/etc/modules-load.d/uinput.conf`)
and the user in the `input` group. Whether presses get through while the
earphones are in headset (mic) mode depends on the earphones.

## The chimes

| sound | meaning |
|---|---|
| rising pair | heard "hey jarvis" - listening |
| falling pair | question sent |
| single soft note | a notification arrived, or a reply came without a voice |
| one low note | heard the wake word but nothing after it |
| low double | something failed (offline, no answer, an error) |
| rising three notes | wake word switched on |
| falling two notes | wake word switched off |

## Service

```
mkdir -p ~/.config/systemd/user
cp systemd/reactor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now reactor
journalctl --user -u reactor -f
```