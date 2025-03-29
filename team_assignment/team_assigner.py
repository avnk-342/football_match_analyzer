from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # Player_id: team_color_id (1 || 2)

    def get_kmeans_model(self, image):
        image_2d = image.reshape(-1, 3)

        #perform kmeans with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans  

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]

        #getting clusturing model
        kmeans = self.get_kmeans_model(top_half_image)

        #get clustur label for each pixel
        labels = kmeans.labels_
        #reshape the labels to the original image shape
        clustured_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        #get the player cluster
        corner_cluster = [clustured_image[0, 0], clustured_image[0, -1], clustured_image[-1, 0], clustured_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_clustor = 1-non_player_cluster

        player_color = kmeans.cluster_centers_[player_clustor]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        for _,player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox) 
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmean = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmean.predict(player_color.reshape(1,-1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id
        return team_id