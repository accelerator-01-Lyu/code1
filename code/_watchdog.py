"""_watchdog.py — memory guard using ctypes (no psutil)."""
import os, time, ctypes
LOG=r"D:\AL_FuzzyClustering_Paper\v2_unified\code\results\watchdog.log"
class MS(ctypes.Structure):
    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
              ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
              ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
              ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
              ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
while True:
    m=MS(); m.dwLength=ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    avail=m.ullAvailPhys/1e9
    with open(LOG,"a",encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] avail={avail:.2f}GB\n")
    time.sleep(1800)
