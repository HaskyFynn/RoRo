# Software Verification

Verified locally on Windows 11 with Python 3.13.14. Exact dependency versions are in `requirements-tested.txt`; the normal installer uses bounded compatible ranges.

- Actual bundled model: SHA256 checked; float32 NCHW input and `[1,300,6]` output validated; real CPU inference completed. A blank image produced no predictions at confidence 0.25. This is a software check, not tag accuracy evidence.
- Original firmware: five steady-state green slots checked for each user. User 1 mismatch documented. No Arduino compiler/hardware available for compile, flash or timing validation.
- Regression checks: wrong wearer assignments despite all labels being present; duplicate detections; misses vs wrong IDs; unavailable model; region overlap; QR payload equality; LED diagnostic; channel ordering/letterbox inversion; NMS and invalid outputs; lossless snapshot pixels; interrupted-log recovery; reports, confusion plots and comparison CSV.
- Real Tk GUI rehearsal: static demo image, timed runs in both experiment modes, distinct analyzed frame IDs, saved snapshots and generated HTML reports. Layout bounds checked at 1260x820 and 900x650. Desktop screenshot capture is unavailable in this tool environment, so pixel-level GUI appearance remains unverified. Static demo imagery and example plot output were visually inspected.
- Software-only example plots and rehearsal runs are stored outside the upload archive in ignored verification/demo folders. They are not experiment results.

CPU benchmark on this machine, three blank-image invocations each: median 1.869 s at one thread, 1.605 s at two, 1.216 s at four, 1.192 s at eight. The receiver uses up to four CPU threads to leave capacity for capture/UI. A compiled CPU runtime probe did not improve throughput and was not adopted. These are local software timings, not hardware system performance results; run `python benchmark.py` on the experiment laptop.

Still requiring physical verification: webcam manual exposure/gain and frame delivery, stripe visibility and model accuracy at the selected distance, emitter polarity and timing, and actual simultaneous wearer trials. A wider webcam view alone does not establish compatibility with the mobile-trained model. Start with the individual-tag pilot and readable QR baseline in the checklist.
