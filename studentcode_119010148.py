import numpy as np

v = 0.93  # control balance between video quality and fluency
gamma = 1  # smooth factor to reduce bitrate rapid fluctuation
p = 5  # buffer influence factor for bitrate choose
chunk_duration = 2
k = 0.01
S_0 = 1000

def student_entrypoint(measured_bandwidth, previous_throughput, buffer_occupancy, available_bitrates, video_time, chunk, rebuffering_time, preferred_bitrate):
    # use available_bitrates to store available bitrate value
    bitrates = []
    for key in available_bitrates:
        bitrates.append((key, available_bitrates[key]))
        
    # rank bitrate from high to low
    for i in range(len(bitrates)):
        for j in range(i + 1, len(bitrates)):
            if bitrates[i][1] < bitrates[j][1]: 
                bitrates[i], bitrates[j] = bitrates[j], bitrates[i]
                
    best_bitrate = choose_optimal_bitrate(bitrates, buffer_occupancy)
    if best_bitrate is None:
        best_bitrate = bitrates[-1][0] # use default bitrate (lowest)
    return best_bitrate # the id of final chosen bitrate

# calculate the score of each bitrate to choose the best bitrate
def choose_optimal_bitrate(bitrates, buffer_info):
    buffer_chunks = buffer_info["time"] / chunk_duration
    bitrate_scores = []
    
    # check if it satisfy no-download option
    no_download = True
    for bitrate_id, bitrate_value in bitrates:
        utility = 1 / (1 + np.exp(-k * (bitrate_value - S_0))) # use sigmoid utility function
        threshold = v * (utility + gamma * p)
        if buffer_chunks <= threshold:
            no_download = False # not satisfy condition
            break
    
    if no_download:
        return None

    for bitrate_id, bitrate_value in bitrates:
        utility = 1 / (1 + np.exp(-k * (bitrate_value - S_0))) 
        score = (v * utility +v * gamma * p - buffer_chunks) / bitrate_value
        print(f"Bitrate ID: {bitrate_id}, Score: {score}")  
        bitrate_scores.append((bitrate_id, score))

    # find the highest bitrate
    best_bitrate_id = bitrate_scores[0][0]  
    highest_score = bitrate_scores[0][1] 
    for bitrate_id, score in bitrate_scores:
        if score > highest_score:           
            best_bitrate_id = bitrate_id
            highest_score = score
    return best_bitrate_id