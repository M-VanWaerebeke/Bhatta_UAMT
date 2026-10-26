# Third-party notices

The root `LICENSE` is the existing MIT notice. It does not replace a different license notice embedded in a source file.

- `ramps.py` and `utils/ramps.py` retain their original Creative Commons Attribution-NonCommercial 4.0 notice. These files have a noncommercial restriction and should not be described as uniformly MIT-licensed with the rest of the repository.
- `utils/util.py` retains its original source-code license header referring to a root license file. The upstream source and applicable original root license have not been recovered during this maintenance update; its provenance needs clarification before relying on it for redistribution.
- `medicalDataLoader.py` credits MedicalZooPytorch for intensity normalization. Loss and training routines also retain research references and comments identifying adaptations from the mean-teacher/UAMT implementations.

The maintenance work preserves these headers and does not claim sole authorship or relicense adapted code. The supported synthetic example uses `UNet3D.py` and `losses.py`; the full historical trainer also imports the ramp utilities above.
