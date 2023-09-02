import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from flask import render_template
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from flask import send_file
import os
import matplotlib.colors as mcolors
import pickle

if not os.path.exists('static'):
    os.makedirs('static')

if not os.path.exists('static/output_folder'):
    os.makedirs('static/output_folder')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logging.basicConfig(level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# G = ox.graph_from_place('Rijeka, Croatia', network_type='drive', simplify=False)  # simplify False gives more nodes
with open('graph_from_place.pkl', 'rb') as f:
    G = pickle.load(f)



matrix_image_path = os.path.join(os.getcwd(), 'static', 'output_folder', 'matrix_image.png')
contour_image_path = os.path.join(os.getcwd(), 'static', 'output_folder', 'contour_image.png')

if os.path.exists(matrix_image_path):
    app.logger.debug("deleting existing file")
    os.remove(matrix_image_path)


def calculate_le_lt(point1, point2):
    node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
    node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])

    road_distance = nx.shortest_path_length(G, node1, node2, weight='length')
    air_distance = geodesic(point1, point2).meters

    if road_distance < 1e-6:
        return float(-1)  # return -1 to indicate undefined

    return air_distance / road_distance


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def home():
    return render_template('map.html')


def calculate(point1, point2):
    if point1 == point2:
        return 0, 0, [], 1  # Le/Lt is 1 when the two points are the same

    # app.logger.debug(f"point1: {point1}, point2: {point2}")

    # Get the nearest nodes to the clicked points
    node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
    node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])
    # app.logger.debug(f"node1: {node1}, node2: {node2}")

    # if node1 in G and node2 in G:
    #     app.logger.debug("Both nodes exist in the graph.")
    # else:
    #     app.logger.debug("One or both nodes are missing.")
    #
    # if nx.has_path(G, node1, node2):
    #     app.logger.debug("A path exists between the nodes.")
    # else:
    #     app.logger.debug("No path exists between the nodes.")

    # Use NetworkX to calculate the shortest path
    shortest_path_nodes = nx.shortest_path(G, node1, node2)
    # app.logger.debug(f"shortest_path_nodes: {shortest_path_nodes}")

    # try:
    #     path = nx.shortest_path(G, node1, node2)
    #     print("Shortest path:", path)
    # except Exception as e:
    #     print("Error:", e)
    #
    # try:
    #     path_length = nx.shortest_path_length(G, node1, node2, weight='length')
    #     print("Shortest path length:", path_length)
    # except Exception as e:
    #     print("Error:", e)

    # Create the shortest path line coordinates
    shortest_path_line = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_path_nodes]
    # app.logger.debug(f"shortest_path_line: {shortest_path_line}")

    # Calculate the road distance
    road_distance = nx.shortest_path_length(G, node1, node2, weight='length')
    # app.logger.debug(f"road_distance: {road_distance}")
    if road_distance < 1e-6:
        return 0, 0, [], 1  # Le/Lt is 1 when the two points are the same

    # Calculate the air distance
    air_distance = geodesic(point1, point2).meters
    # app.logger.debug(f"air_distance: {air_distance}")

    if road_distance < air_distance:
        print('NOT POSSIBLE ROAD DISTANCE')
        return

    # Calculate the Le/Lt factor
    le_lt_factor = air_distance / road_distance
    # app.logger.debug(f'The Le/Lt factor is {le_lt_factor}.')

    # test_values
    # air_distance = 9556.785333266234
    # road_distance = 7421.784999999998
    # shortest_path_line = [(45.326749, 14.4410185), (45.3268541, 14.4407581), (45.3271253, 14.4400859), (45.3271367, 14.4400563), (45.3274117, 14.439301), (45.3274464, 14.4392145), (45.3275974, 14.4388359), (45.3277001, 14.4385135), (45.3277325, 14.4384071), (45.3280654, 14.4373162), (45.3280939, 14.4372232), (45.3281314, 14.4371157), (45.3282773, 14.4367728), (45.328387, 14.4364647), (45.3294604, 14.4344848), (45.3296605, 14.4340867), (45.3297745, 14.4338694), (45.329837, 14.433732), (45.3298584, 14.4336847), (45.3299, 14.43359), (45.3299227, 14.4335242), (45.3299455, 14.4334422), (45.3300924, 14.4328505), (45.3301381, 14.432675), (45.3301817, 14.4325066), (45.3302249, 14.4323231), (45.3303115, 14.4319806), (45.3303288, 14.4319061), (45.3303461, 14.4318274), (45.3303643, 14.431732), (45.3303816, 14.4316363), (45.3305216, 14.4306985), (45.3305377, 14.4305951), (45.3305586, 14.4304538), (45.3305966, 14.4302093), (45.330654, 14.4298303), (45.3306962, 14.4295727), (45.3307101, 14.4294884), (45.3307372, 14.4293391), (45.3307534, 14.4292582), (45.3308409, 14.4288088), (45.3308639, 14.4286925), (45.3308824, 14.428578), (45.331062, 14.4274083), (45.3310836, 14.4272995), (45.3311131, 14.4271933), (45.3312271, 14.4268215), (45.3314567, 14.4261454), (45.331599, 14.4257509), (45.3316137, 14.4257101), (45.3316883, 14.4255034), (45.3317536, 14.4253466), (45.3318365, 14.4251763), (45.3319867, 14.4249356), (45.3321943, 14.4246588), (45.3328142, 14.423897), (45.3333248, 14.4232976), (45.3334928, 14.423062), (45.3336963, 14.4227486), (45.3337837, 14.4226143), (45.3339616, 14.4223522), (45.3340787, 14.4221289), (45.3341303, 14.4220235), (45.3343655, 14.4214259), (45.3344762, 14.4211445), (45.3348099, 14.4201799), (45.3349627, 14.4197768), (45.3352976, 14.4188623), (45.335419, 14.4185001), (45.3355166, 14.41822), (45.335588, 14.4180306), (45.3356693, 14.4178063), (45.3359647, 14.4170285), (45.3361383, 14.4165218), (45.3363842, 14.415889), (45.33654, 14.4155725), (45.3366766, 14.4153192), (45.3371455, 14.4146433), (45.3373691, 14.4143838), (45.337539, 14.4141725), (45.3377122, 14.4138466), (45.3377976, 14.4136623), (45.337871, 14.4134475), (45.3379459, 14.4131624), (45.3379983, 14.4128684), (45.3380196, 14.4126261), (45.3380316, 14.4123014), (45.3380542, 14.4116245), (45.3380602, 14.4114437), (45.3380703, 14.4110623), (45.3380715, 14.4110159), (45.3380683, 14.4108009), (45.338068, 14.4106039), (45.3380806, 14.4099728), (45.3381165, 14.4091724), (45.33813, 14.40893), (45.3381349, 14.4087188), (45.3381718, 14.4077658), (45.3381985, 14.406972), (45.3382058, 14.4067557), (45.3382107, 14.4066088), (45.3382204, 14.4063955), (45.3382515, 14.4059347), (45.338274, 14.4057167), (45.3383029, 14.4054806), (45.3383743, 14.4050164), (45.338417, 14.4047502), (45.3384968, 14.4042523), (45.338575, 14.4037394), (45.3385962, 14.4035498), (45.3386474, 14.4029618), (45.3386579, 14.4028421), (45.3386622, 14.4027917), (45.3386865, 14.4025131), (45.3387174, 14.4021584), (45.338743, 14.401897), (45.3387618, 14.4014533), (45.3387666, 14.4010928), (45.3387485, 14.4000113), (45.3387293, 14.3995046), (45.3387169, 14.3989997), (45.3387178, 14.3988139), (45.3387356, 14.3986099), (45.3388089, 14.3982455), (45.3389403, 14.3978388), (45.3390931, 14.397367), (45.3392119, 14.3969692), (45.3392482, 14.3967959), (45.3392872, 14.3966959), (45.3393087, 14.3965562), (45.3393604, 14.3960622), (45.3394101, 14.3952458), (45.3393893, 14.3950402), (45.3394603, 14.3942153), (45.3395773, 14.3927728), (45.3395852, 14.3926704), (45.3395915, 14.3925881), (45.3396008, 14.3924679), (45.3396335, 14.3920438), (45.3396767, 14.3914827), (45.339845, 14.3892307), (45.3398561, 14.3890825), (45.3399606, 14.3876449), (45.3399773, 14.3871816), (45.3399653, 14.3869235), (45.339938, 14.3866363), (45.3399148, 14.3863097), (45.3398642, 14.3860036), (45.3398581, 14.3859668), (45.339784, 14.3856621), (45.3395279, 14.384511), (45.3394574, 14.3842674), (45.3394256, 14.384111), (45.3393712, 14.383836), (45.3393613, 14.3835692), (45.3393787, 14.3833176), (45.3394046, 14.3831233), (45.3396465, 14.3822118), (45.3397872, 14.3816803), (45.3398967, 14.3813216), (45.3399062, 14.3812903), (45.339922, 14.3811879), (45.3399364, 14.3811), (45.3399485, 14.3810285), (45.3399764, 14.3808933), (45.3399901, 14.3808324), (45.3400266, 14.3806574), (45.3400563, 14.3804275), (45.3400756, 14.380205), (45.3400877, 14.37936), (45.3400865, 14.3786181), (45.3400886, 14.3785073), (45.3400858, 14.3783443), (45.3400884, 14.378185), (45.3400963, 14.3780473), (45.3401052, 14.3779564), (45.3401296, 14.3777973), (45.3401615, 14.3776636), (45.3401989, 14.3775338), (45.3402688, 14.3773604), (45.34036, 14.3771912), (45.3404981, 14.377004), (45.3405733, 14.376879), (45.3406457, 14.3767335), (45.3406877, 14.3766093), (45.3407288, 14.376499), (45.3408189, 14.3761734), (45.3408664, 14.3759761), (45.3412805, 14.3742564), (45.3413651, 14.3738728), (45.3414612, 14.3734371), (45.3415223, 14.3731812), (45.3415357, 14.3731288), (45.3416721, 14.3725822), (45.3417119, 14.3724043), (45.3417584, 14.3721964), (45.3417763, 14.3721164), (45.3418999, 14.3715715), (45.3419218, 14.3714101), (45.3419922, 14.3708925), (45.3420066, 14.3706587), (45.3420303, 14.3702159), (45.3420275, 14.3697002), (45.3420259, 14.3695246), (45.3420248, 14.3693947), (45.3420231, 14.3692053), (45.3420306, 14.3687653), (45.3420388, 14.368438), (45.3420632, 14.368127), (45.3421341, 14.3675924), (45.3422227, 14.3671795), (45.3424012, 14.3663885), (45.3426079, 14.3654013), (45.3427186, 14.3648952), (45.342775, 14.3646292), (45.3427807, 14.3646023), (45.342863, 14.3643176), (45.3429665, 14.3640417), (45.3431849, 14.3636861), (45.3433699, 14.3633759), (45.3434898, 14.3631411), (45.343551, 14.3629793), (45.3435882, 14.3627866), (45.3436016, 14.3625915), (45.3435797, 14.3623692), (45.3435074, 14.3620123), (45.3434381, 14.3614696), (45.3434197, 14.3610757), (45.3434204, 14.3608856), (45.3434375, 14.3606941), (45.3434887, 14.3604679), (45.3435583, 14.3602679), (45.3437085, 14.3599348), (45.3438325, 14.3596709), (45.3442935, 14.3588295), (45.3445846, 14.3582909), (45.3450742, 14.3573614), (45.3453852, 14.3567987), (45.345703, 14.356214), (45.3456559, 14.3561584), (45.3459349, 14.3556282), (45.3460026, 14.3554305), (45.3460382, 14.3552579), (45.3460169, 14.3551306), (45.3459649, 14.3550432), (45.345818, 14.3549275), (45.3457393, 14.3548894), (45.3455328, 14.3547438), (45.3454317, 14.3546702), (45.3453771, 14.3545591), (45.345338, 14.3544369), (45.3453334, 14.3543567)]
    # le_lt_factor = 0.1

    return air_distance, road_distance, shortest_path_line, le_lt_factor


@app.route('/get_matrix_image')
def get_matrix_image():
    # return send_file(matrix_image_path, mimetype='image/png')
    app.logger.debug("Matrix image requested")
    return send_from_directory("static/output_folder", "matrix_image.png")


@app.route('/calculate_le_lt_matrix', methods=['POST'])
def calculate_le_lt_matrix():
    data = request.get_json()
    points = data['points']

    n = len(points)
    matrix = np.zeros((n, n))
    paths = []

    for i in range(n):
        for j in range(n):
            if i != j:
                # Using the calculate function to get air and road distances and le_lt factor
                air_distance, road_distance, road_path, le_lt = calculate(points[i], points[j])
                matrix[i][j] = le_lt

                paths.append({
                    "start": i,
                    "end": j,
                    "path": road_path,
                    "air_distance": air_distance,
                    "road_distance": road_distance
                })

    colors = ["red", "orange", "green", "blue"]
    boundaries = [0, 0.25, 0.5, 0.75, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    sns.heatmap(matrix, annot=True, cmap=cmap, norm=norm)

    # plt.show()

    # Save the image
    if os.path.exists(matrix_image_path):
        app.logger.debug("deleting existing file")
        os.remove(matrix_image_path)

    plt.savefig(matrix_image_path)

    # Close the plot
    plt.close()

    app.logger.debug(f"calculated matrix: {matrix}")

    return jsonify({
        'matrix': matrix.tolist(),
        'paths': paths,
    })

@app.route('/contour_map', methods=['POST'])
def generate_contour_map():
    # Extract marker data from the post request
    data = request.json
    latitudes = [point[0] for point in data['markers']]
    longitudes = [point[1] for point in data['markers']]

    print(latitudes, longitudes)

    # Define Korzo's coordinates
    korzo_coords = [45.32715328765221, 14.441230595111849]
    latitudes.append(korzo_coords[0])
    longitudes.append(korzo_coords[1])

    # Calculate Le/Lt coefficients in relation to Korzo for each marker including Korzo
    le_lt_values = [calculate(korzo_coords, (lat, lon))[-1] for lat, lon in zip(latitudes, longitudes)]

    # Generate contour map
    plt.tricontourf(longitudes, latitudes, le_lt_values, 20)
    plt.colorbar(label='Le/Lt Coefficient')

    # Plotting the markers
    plt.scatter(longitudes[:-1], latitudes[:-1], color='red', s=10)

    # Adding Korzo point and label it
    plt.scatter(korzo_coords[1], korzo_coords[0], color='blue', s=40, marker='o')
    plt.text(korzo_coords[1], korzo_coords[0] + 0.001, 'Korzo', horizontalalignment='center')  # Label for Korzo

    plt.title('Contour map of Le/Lt coefficients')

    # Save to a temporary file and send this as response
    # temp_filename = "temp_contour_plot.png"
    plt.savefig(contour_image_path)
    plt.close()

    return send_file(contour_image_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))