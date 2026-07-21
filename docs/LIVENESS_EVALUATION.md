# Attendee Liveness Evaluation

Issue #21 adds a basic randomized head-turn check to the public attendee flow. It
is a local deterrent against static-photo submissions, not production-grade identity
assurance. Complete this evaluation on the target Windows kiosk before marking the issue
done.

## Setup

1. Start the Flask app with `python -m src.web` and open the attendee interface in
   current Chrome and Edge releases.
2. Use an indexed test event and obtain informed consent from every genuine participant.
3. Do not retain camera frames, screenshots, names, or embeddings. Record only the
   anonymous attempt type, browser, lighting condition, requested direction, and result.

## Test Matrix

Run at least 10 genuine attempts across different participants, both challenge
directions, ordinary indoor lighting, dim lighting, glasses where applicable, and small
pose variations. Then run at least 10 presentation-attack attempts using printed photos
and photos displayed on another screen. Include camera denial and a browser/device with
no available camera.

For each attempt record:

| Attempt | Type | Browser | Condition | Direction | Expected | Actual | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Genuine / print / screen | Chrome / Edge | Lighting or device | Left / right | Pass / reject | Pass / reject | Anonymous observations only |

## Acceptance Review

- Every accepted attendee sequence reaches matching using its final verified frame.
- Static front/front/front sequences and wrong-direction turns are rejected.
- Camera denial and unavailable-camera states direct the attendee to staff lookup.
- No captured frame or query embedding appears in event folders, SQLite, logs, or temp files.
- Document the genuine false-rejection rate and presentation-attack acceptance rate.
- If results are not acceptable, tune the pose constants with a separate consented
  calibration set; do not tune against the final evaluation attempts.

Video replay and synthetic camera injection remain outside this basic check's assurance.
Escalate to a reviewed passive anti-spoofing model or dedicated liveness provider before
using PhotoMatch as a high-assurance identity-verification system.
