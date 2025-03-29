from utils import read_video, save_video
from tracking import Tracker
import cv2
from team_assignment import TeamAssigner

def main():
    #reading video
    video_frames = read_video('./input_vidoes/08fd33_4.mp4')

    


    #Intialise the tracker
    tracker = Tracker("models/best.pt")

    track = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # #getting image of single player
    # for track_id, player in track['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     #crop bbox from frame
    #     cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     #save the cropped frame
    #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_frame)
    #     break
    
    #Assigning player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], track['players'][0])
    
    for frame_number, player_track in enumerate(track['players']):
        for player_id, tracks in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_number], tracks['bbox'], player_id)
            track["players"][frame_number][player_id]['team'] = team
            track["players"][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

        

    # Draw output
    ## Draw object object track
    output_video_frames = tracker.draw_annotations(video_frames, track)

    #saveing the video
    save_video(output_video_frames, 'output_videos/outputVideo.avi')

if __name__ == '__main__':
    main()