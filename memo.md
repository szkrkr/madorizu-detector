source ~/.profile
eval $(/usr/libexec/path_helper -s)
if [[ -s ~/.nvm/vnm.sh ]];
 then source ~/.nvm/nvm.sh
fi
export PATH=$HOME/.nodebrew/current/bin:$PATH
export PATH=$PATH:/Users/SuzukiRikuro/.nodebrew/current/bin

# Setting PATH for Python 2.7
# The original version is saved in .bash_profile.pysave
PATH="/Library/Frameworks/Python.framework/Versions/2.7/bin:${PATH}"
PATH="/Users/SuzukiRikuro/workspace/sdks/flutter/bin:${PATH}"
export PATH
# added by Anaconda3 5.3.1 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
~                                                                                                                                                      
~                                                                                                                                                      
~                              




# ufunc 'invert' not supported for the input types,
https://stackoverflow.com/questions/52432549/invert-pixel-color-between-a-contour-and-the-extended-contour

# only integer scalar arrays can be converted to a scalar index
https://stackoverflow.com/questions/50997928/typeerror-only-integer-scalar-arrays-can-be-converted-to-a-scalar-index-with-1d

* not enough values to unpack (expected 3, got 2)
Traceback (most recent call last):
  File "detector.py", line 116, in <module>
    rooms, colored_house = find_rooms(img.copy())
  File "detector.py", line 51, in find_rooms
    _, contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ValueError: not enough values to unpack (expected 3, got 2)

https://teratail.com/questions/194500


* src.type() == CV_8UC1 in function 'threshold'
Traceback (most recent call last):
  File "detector.py", line 149, in <module>
    find_walls(img)
  File "detector.py", line 118, in find_walls
    retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.error: OpenCV(4.3.0) /Users/travis/build/skvark/opencv-python/opencv/modules/imgproc/src/thresh.cpp:1529: error: (-215:Assertion failed) src.type() == CV_8UC1 in function 'threshold'

-> need to 
