# include OpenCV 3.1 lib
OPENCV_PATH = C:/opencv/build
INCLUDEPATH += $$OPENCV_PATH/include
LIBS += -L$$OPENCV_PATH/x64/vc12/lib/
LIBS += -lopencv_world310

SOURCES += \
    main.cpp
    main.cpp
