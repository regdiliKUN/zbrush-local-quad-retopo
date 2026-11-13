================================================================
  Local Quad Retopo (LocalQuadRetopo) - README       Version 1.4.3
================================================================

This file covers installing, uninstalling and installation
problems. For how to use the plug-in, see "User Guide.txt" in the
same folder.


1. Requirements
----------------------------------------------------------------

  - Windows 10 or Windows 11 (64-bit)
  - ZBrush 2022 (2022.0.5 or later) or ZBrush 2025
  - An internet connection for the first installation

The calculations run in Python. The installer checks whether this
computer has a usable Python and, if not, downloads one for you.
Nothing needs to be prepared by hand.


2. About Python (usually nothing to do)
----------------------------------------------------------------

During installation the installer works through these steps:

  1. If this computer already has Python 3 with NumPy and SciPy,
     that Python is used and nothing is downloaded.
  2. Otherwise a Python for this plug-in only is downloaded to
     C:\LQR\python, together with NumPy and SciPy. The window shows:
         [1/3] Downloading Python ...
         [2/3] Installing pip ...
         [3/3] Downloading NumPy and SciPy ...
     About 60 MB in total; one to a few minutes depending on the
     connection. This happens on the first installation only;
     later upgrades of the plug-in do not download it again.

This private Python is not added to the system path, does not affect
other programs and does not conflict with a Python you installed
yourself. Deleting the C:\LQR folder removes it.

If the main download address cannot be reached, the installer tries
mirror servers automatically.


3. Installing the plug-in
----------------------------------------------------------------

  1. Save your work and quit ZBrush.
  2. Extract the whole zip file and double-click INSTALL.bat.
  3. The installer looks for ZBrush on this computer and lists what
     it finds with numbers. Type a number and press Enter (Enter
     alone picks the first one).
     - Both 2022 and 2025 installed and you want the plug-in in
       both: run INSTALL.bat twice and pick one each time.
     - The one you want is not listed (for example a custom folder
       on drive D): type 0, press Enter, then type the folder that
       contains ZBrush.exe and press Enter, for example:
           D:\ZBrush 2022.0.5
       (Select ZBrush.exe in Explorer, hold Shift, right-click and
        choose "Copy as path", then paste it. With or without
        \ZBrush.exe at the end both work.)
       After one installation this folder is listed automatically.
  4. If needed, the installer downloads Python (see section 2).
  5. When the window ends with the green line
         Installed: Local Quad Retopo 1.4.3
     the installation succeeded. Press any key to close it.
     If a yellow note below it says the panel must be loaded once by
     hand: start ZBrush, choose Zscript > Load and pick the
     LocalQuadRetopo.txt the note names. This is needed only once;
     ZBrush loads the plug-in by itself after that.
  6. Start ZBrush and open "Local Quad Retopo" in the ZPlugin menu.

The installer puts these files in place:
  - ZBrush folder\ZStartup\ZPlugs64\LocalQuadRetopo.txt
    (the plug-in panel)
  - ZBrush folder\ZStartup\ZPlugs64\LocalQuadRetopo.zsc
    (the compiled panel. ZBrush loads only .zsc files at startup,
     so the installer places the copy compiled for 2022 or 2025;
     without one in the package, ZBrush creates it when you load
     the panel once by hand)
  - the folder C:\LQR\ (the calculation programs, the private
    Python and the temporary files of each run)

To reinstall or upgrade, simply run INSTALL.bat again. Replaced
files are backed up to folders starting with backup_ in C:\LQR.

The Chinese and the English edition use the same file names, so only
one of them can be installed in a ZBrush at a time. Installing one
replaces the other.


4. Uninstalling
----------------------------------------------------------------

  1. Quit ZBrush.
  2. Delete LocalQuadRetopo.txt and LocalQuadRetopo.zsc (if present)
     from ZBrush folder\ZStartup\ZPlugs64\.
  3. Delete the folder C:\LQR.


5. Cleaning up temporary files
----------------------------------------------------------------

Every run leaves a few model files in C:\LQR for troubleshooting.
After long use you can delete:
  - all .obj, .log and .claim files
  - the .txt reports whose names start with lqr followed by digits
  - the folders starting with backup_

Do not delete: the .py files, LqrLaunch.exe, the python folder,
lqr_python_path.txt and lqr_zbrush_dirs.txt.
If you delete them by mistake, run INSTALL.bat again to restore them.


6. Installation problems
----------------------------------------------------------------

  * Python, pip or NumPy / SciPy could not be downloaded
    -> Check the network and run INSTALL.bat again; the download
       starts over. Company networks sometimes block downloads; try
       another network.

  * You want to switch to the private Python (for example because
    your own Python has problems)
    -> Type cmd in the address bar of the folder with INSTALL.bat
       and press Enter. In the window that opens, type this line
       and press Enter:
           INSTALL.bat -InstallPython

  * "ZBrush is running"
    -> Save your work and quit ZBrush, then install.

  * "ZBrush.exe was not found"
    -> The folder you typed is wrong. Make sure ZBrush.exe is
       directly inside it.

  * "Access ... denied"
    -> ZBrush installed under C:\Program Files needs administrator
       rights to add a plug-in. Right-click INSTALL.bat and choose
       "Run as administrator".

  * "Local Quad Retopo" is missing from the ZPlugin menu
    -> Check that you installed into the ZBrush you actually use,
       then quit ZBrush completely and start it again.
       If it is still missing, choose Zscript > Load and pick
       ZBrush folder\ZStartup\ZPlugs64\LocalQuadRetopo.txt
       once; it loads by itself after that.

  * A message at the top says the background files are missing
    -> Files in C:\LQR were deleted or blocked by antivirus
       software. Run INSTALL.bat again; if the antivirus software
       blocks LqrLaunch.exe, add C:\LQR to its trusted list.


7. Files in this package
----------------------------------------------------------------

  INSTALL.bat              the installer; double-click it
  Install.ps1              the installer script itself
  LocalQuadRetopo.txt      the ZBrush plug-in panel
  LocalQuadRetopo_2022.zsc the panel compiled for ZBrush 2022 (if included)
  LocalQuadRetopo_2025.zsc the panel compiled for ZBrush 2025 (if included)
  LqrLaunch.exe            starts the calculation in the background
                           without a command window
  lqr_run_pipeline.py      entry point of the calculation
  lqr_recover.py           main retopology steps
  lqr_validate.py          result checks
  recovery_geometry.py     quad pairing and refinement
  patch_remesh.py          honeycomb layout
  core_patch.py            regular layout
  README.txt               this file
  User Guide.txt           how to use the plug-in
