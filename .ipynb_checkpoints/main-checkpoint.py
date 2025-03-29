from utils import read_video, save_video
from tracking import Tracker

def main():
    #reading video
    video_frames = read_video('./input_vidoes/08fd33_4.mp4')

    #Intialise the tracker
    tracker = Tracker("models/best.pt")

    track = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Draw output
    ## Draw object object track
    output_video_frames = tracker.draw_annotations(video_frames, track)

    #saveing the video
    save_video(output_video_frames, 'output_videos/outputVideo.avi')

if __name__ == '__main__':
    main()