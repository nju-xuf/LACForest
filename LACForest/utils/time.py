from datetime import datetime


def ctime():
    # A formatter on current time used for printing running status.
    ctime = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    return ctime
