Overview:
    This is a computer vision powered tennis coaching app that is made to analyze videos of tennis swings and provide 
    free feedback. All aspects of the app are free, and it provides coaching on serves, forehands, one-handed backhands, 
    and two-handed backhands. It utilizes a pretrained pose detection model from Ultralytics YOLO, as well as a racket
    and ball tracking model. Both of these models combined with pre-programmed feedback loops allows it to provide
    feedback based on specific swing metrics and measurements.

Instructions:
    To use this codebase, first film yourself swinging, either a forehand, backhand, or serve. The app works best if you
    set up a tripod, or have someone record you swinging from the fence of the court. Only upload one swing at a time.
    After you have the video, upload it to your computer. If you have a mac, then you can go to the video, then click
    file in the top right. After that, click export, and then export 1 unmodified original. This will then put the video
    in your files. After that, all you have to do is change the filepath on the code to your filepath, and add the name
    of the video file. The last step is to input whether you are left or right-handed by assigning the "Hand" variable,
    and then you are good to go.

    You will need to download the pretrained YOLO Pose estimation model, which can be found here: https://docs.ultralytics.com/tasks/pose/
    You will then need to specify the path to this model in a command line argument
