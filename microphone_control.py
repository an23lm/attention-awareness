import os
import logging


class MicrophoneControl:
    is_muted = False

    @staticmethod
    def mute():
        if (not MicrophoneControl.is_muted):
            MicrophoneControl.is_muted = True
            logging.info("Mute Mic")
            os.system('osascript -e "set volume input volume 0"')

    @staticmethod
    def unmute():
        if (MicrophoneControl.is_muted):
            MicrophoneControl.is_muted = False
            logging.info("Unmute Mic")
            os.system('osascript -e "set volume input volume 100"')
