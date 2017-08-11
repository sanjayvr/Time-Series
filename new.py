from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.anomaly import Anomaly
import time

if __name__ == '__main__':

	my_detector = AnomalyDetector('m.csv')
	score = my_detector.get_all_scores()
	anomalies = my_detector.get_anomalies()

	for i in anomalies:
		print('Sanjay')

	anom_score = []

	for (timestamp,value) in score.iteritems():
    		t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    		anom_score.append([t_str, value])

	print(anom_score)
