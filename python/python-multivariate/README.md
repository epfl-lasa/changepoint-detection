Multivariate Online Changepoint Detection:

class Detector.py : performs detection algorithm on current datum; stores variables for detection
	detect			-- performs algorithm on current datum
	retrieve		-- returns values of hyperparameters (theta), changepoints (CP) and runlength (maxes) after reading all data

class StudentTMulti : methods connected to NIW prior, called by Detector; stores variable of prior
	pdf			-- returns predictive probability of current datum 
	update_theta		-- updates hyperparameters after each 
	curr_theta, save_theta	-- saves hyperparameters of last segment
	reset_theta		-- resets hyperparameters to consider current segment only
	retrieve_theta		-- returns values of hyperparameters

hazards.py : constant hazard function


To try it out, run example_stream_data.py
