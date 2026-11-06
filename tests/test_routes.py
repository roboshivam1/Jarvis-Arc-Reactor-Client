import json

from reactor.routes import best_profile, bluetooth_cards, pick, short_names

SOURCES = ("53\talsa_output.platform-bcm2835_audio.stereo-fallback.monitor\tPipeWire\ts16le 2ch 48000Hz\tSUSPENDED\n"
           "77\tbluez_output.F4_C2_EF_45_E8_DD.1.monitor\tPipeWire\ts16le 1ch 16000Hz\tSUSPENDED\n"
           "78\tbluez_input.F4_C2_EF_45_E8_DD.0\tPipeWire\ts16le 1ch 16000Hz\tRUNNING\n")
SINKS = ("53\talsa_output.platform-bcm2835_audio.stereo-fallback\tPipeWire\ts16le 2ch 48000Hz\tSUSPENDED\n"
         "80\talsa_output.usb-C-Media_USB_Audio-00.analog-stereo\tPipeWire\ts16le 2ch 48000Hz\tIDLE\n"
         "77\tbluez_output.F4_C2_EF_45_E8_DD.1\tPipeWire\ts16le 1ch 16000Hz\tRUNNING\n")


def test_bluetooth_first_and_never_a_monitor() -> None:
    assert pick(short_names(SOURCES), "auto") == "bluez_input.F4_C2_EF_45_E8_DD.0"
    assert pick(short_names(SINKS), "auto") == "bluez_output.F4_C2_EF_45_E8_DD.1"
    only_monitors = short_names(SOURCES)[:2]
    assert pick(only_monitors, "auto") is None          # never listen to our own output
    no_bt = [n for n in short_names(SINKS) if not n.startswith("bluez")]
    assert pick(no_bt, "auto").startswith("alsa_output.usb")
    assert pick(short_names(SINKS), "alsa_output.platform-bcm2835_audio.stereo-fallback") \
        == "alsa_output.platform-bcm2835_audio.stereo-fallback"
    assert pick(short_names(SINKS), "gone") is None


CARDS = json.dumps([
    {"index": 1, "name": "alsa_card.platform-bcm2835_audio", "active_profile": "output:stereo-fallback",
     "profiles": {}},
    {"index": 60, "name": "bluez_card.F4_C2_EF_45_E8_DD", "active_profile": "a2dp-sink",
     "profiles": {
         "off": {"description": "Off", "sinks": 0, "sources": 0, "priority": 0, "available": True},
         "a2dp-sink-sbc": {"sinks": 1, "sources": 0, "priority": 18, "available": True},
         "a2dp-sink": {"sinks": 1, "sources": 0, "priority": 19, "available": True},
         "headset-head-unit-cvsd": {"sinks": 1, "sources": 1, "priority": 1, "available": True},
         "headset-head-unit": {"sinks": 1, "sources": 1, "priority": 2, "available": True},
     }},
])


def test_earphones_go_to_the_best_mode() -> None:
    [card] = bluetooth_cards(CARDS)
    assert card.active == "a2dp-sink"
    assert best_profile(card, headset=True) == "headset-head-unit"
    assert best_profile(card, headset=False) == "a2dp-sink"
