# oculus_reader APK

`teleop-debug.apk` is the unmodified Quest app from
[rail-berkeley/oculus_reader](https://github.com/rail-berkeley/oculus_reader)
(Apache-2.0). It streams controller poses and button states to logcat, which
`raiden/robot/oculus.py` reads over ADB. It is installed on the headset
automatically the first time `rd teleop --control oculus` runs.
