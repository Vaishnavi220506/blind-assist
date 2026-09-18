# ToF observable contract: capability boundaries and migration target

Current executable source is mz115_zonal_tof.py:8x8,45deg total FOV,9 private
subrays per zone, at most2 returned targets,50mm histogram bins,600mm peak
separation,40mm synthetic Gaussian range noise. A slot becomes SIM_MERGED for
multiple joined peaks OR private hit span>=100mm. Thus MERGED does not imply
multiple physical surfaces. Public slots expose distance/sigma/strength/status;
histogram and hit identities remain evaluator-only. This is a hypothetical
forward model, not calibrated ST firmware or measured photon transport.

The user supplies official references for CX up to4 targets (default1), approx
600mm separation, and CH CNH up to64zones/128bins with a64zone/18bin/15fps example:
- https://www.st.com/content/st_com/en/technical-documents/UM3109.html
- https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html

Verification attempted2026-09-18: ST product/PDF requests returned a7350-byte
HTML verification page, not document bytes; unauthenticated official GitHub
API requests were rate-limited. These exact figures remain user-supplied,
not independently verified in this task. Saved responses are under
artifacts.local/work/tof-dither-20260918/st-sources/. Do not label them PDFs
successfully inspected or use them as calibrated simulation parameters.

2026-09-19 correction from the user supplies the official UM3109/PDF and
VL53L8CH datasheet/product references and the intended readings: CX up to4
targets per zone (default1; Closest/Strongest ordering; approximately600mm
separation for separate detection); CH up to64 zones and128 histogram bins,
with output-buffer/bandwidth examples such as8x128 at20Hz,32x36 at15Hz and
64x18 at15Hz. These remain user-provided official-source claims in this local
record because the direct HTTP fetch above was blocked; they are not silently
promoted to independently inspected bytes or calibrated simulator parameters.

Future opt-in versioned observation contract should preserve configured zone
geometry, reported target count/order, independent per-target distance/quality,
and (when supplied by driver) histogram bin origin/width/count, per-zone/bin
signal, integration time and acquisition timestamps. Include actual sensor
part/firmware/mode and separate driver status from simulation labels. Missing
histogram is absent/UNKNOWN, never synthesized from evaluator private bins.
Keep configurations and exposure/frame rate explicit; changing them changes the
observation budget. Existing frozen raw input format/defaults remain reproducible.

No runtime sensor-contract migration was performed: official field/unit semantics
are not verified and no real driver packet was supplied. This document is a
migration target, not a tested CX/CH adapter. Multi-target/CNH resolve depth-mode
ambiguity, not intra-zone angular position by themselves. A future histogram
simulation must have its own public output provenance; private nine-ray histogram
is not a real CNH measurement and must not silently enter historical baselines.
MZ182 deliberately keeps current ToF output and tests angular sampling only.
