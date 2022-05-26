# ephys-imaging_correlation
Quick correlation script to align whole-cell and cell-attached patch clamp data to 2-photon imaging data from ScanImage. 

Trigger sync performed on digitizer so only the timing offset is needed. 
Framerates can be obtained from ScanImage usr files.

Input is .mat files containing raw data in the format: "unnamed" = imaging data, "unnamed1" = patch data, and "unnamed2" = reference imaging data, if present.

For Ji lab collaboration; both a python and matlab script are present for ease of use.
