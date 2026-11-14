****(*/ω＼*)****
**ᓚᘏᗢ**
Team lead here

---

# Animal deterrent – how the pieces fit

A camera spots an animal, the device decides whether it is a threat, responds with a
deterrent (or deliberately with nothing), keeps checking until the animal is gone, and
learns from the result.

```
motion -> look (burst of photos) -> which animal, which way?      vision/detector.py
       -> threat? which response?                                  decision_engine.py
       -> respond (printed for now; Arduino later)                 run.py: Hardware
       -> keep looking until it is gone / in the area / time's up  run.py: Pipeline.follow_up
       -> one record per encounter                                 data_collection.py
       -> learn from it                                            decision_engine.py
```

## Try it

```bash
pip install -r vision/requirements-pc.txt

python run.py --simulate 20        # made-up world: no camera, model or Arduino needed
python test.py                     # what the engine has learned + all checks
python run.py --camera 0 --model yolo26n.pt --alias person=deer
                                   # no minis yet: stock model, and you play the deer.
                                   # Walk toward the right of the view, then leave (or don't).
python run.py --camera 0           # the real thing, once vision/models/ has your trained model
```

Choosing what to protect when you deploy:

```bash
python run.py --camera 0 --scenario livestock       # another scenario from settings.yaml
python run.py --camera 0 --settings site_b.yaml     # or a whole settings file per site
```

Responses are only printed for now (`[hardware] would fire: sound`). The `Hardware` class
in `run.py` is the one place to connect the Arduino.

## Adapting it to a site: `settings.yaml`

| Setting | What it does |
|---|---|
| `scenario` | `crops` (deter deer), `livestock` (deter fox, wolf, bear) or `road` (deter all). Add your own under `scenarios`. |
| `critical_area` | Which side of the camera view the protected area is on, and how much of the view it covers. |
| `min_confidence` | How sure the detector must be before the device reacts. |
| `deterrents` | What the device can respond with. "none" is always an option. |
| `learning` | Share of threats kept as control (no deterrent), and how fast old results fade. |
| `follow_up` | How often to look again after responding, when to give up, how many empty looks mean "gone", and how weak a sighting still counts while following an animal. |

The animals themselves are listed once, in `vision/config.yaml`.

## How a threat is decided and judged

- **Threat** = an animal on the scenario's list, detected with enough confidence, that is
  moving toward the protected area or already inside it. Everything else is logged and left alone.
- **Response** = 1 in 10 threats gets nothing on purpose (control). Otherwise the engine picks
  between none and each deterrent, favouring what has worked on that animal but still
  re-testing the others, so it adapts when animals get used to something.
- **Outcome**, from looking again until the animal is gone or 20 s have passed:
  `left` (gone, away from the area: worked), `entered` (went into the area: didn't),
  `stayed` (still in view at the end: didn't), `unknown` (lost sight of it: not counted).

## Known limits

- "Gone" means the detector found no animal in two looks in a row. A detector that loses an
  animal that is still standing there will report it as gone. While following an animal the
  loop already accepts weaker sightings (`follow_up.min_confidence`); if it still happens with
  your trained model, raise `follow_up.empty_looks_before_gone`.
- `seconds_until_gone` is measured up to the first look that no longer found the animal, so it
  is only as fine as the pause between looks (about 2 s with the default settings).
- Only one animal is followed at a time: the threat seen most clearly.
- An animal still in view after 20 s counts as `stayed`, even if it is slowly walking away.
- Timing on the UNO Q itself is untested. Each look costs three detections there.
- `decision_engine.choose_deterrent()` now takes the animal's `direction` and `position`
  instead of `approach`. Code that still calls it the old way gets an error, not a wrong answer.

## Files

| File | What it is |
|---|---|
| `settings.yaml`, `settings.py` | Behaviour of the device, and its loader |
| `decision_engine.py` | Threat check, choice of response, learning |
| `data_collection.py` | The encounter record and the log (`encounters.jsonl`) |
| `run.py` | The loop. `Hardware` in it is where the Arduino gets connected |
| `simulation.py` | Made-up animals for `run.py --simulate` and the mock data |
| `test.py` | Checks for all of the above |
| `encounters_mock.jsonl`, `mock_data/` | Invented encounters and the script that makes them |
| `vision/` | Camera, dataset collection, training, detector (see `vision/README.md`) |
