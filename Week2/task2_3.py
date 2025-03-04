import sys
import trackeval  # Asegúrate de que TrackEval está en tu PYTHONPATH

def evaluate_tracker(tracker_name):
    """ Evalúa un tracker en MOTChallenge con IDF1 y HOTA """
    eval_config = {
        "METRICS": ["HOTA", "IDF1"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "TRACKERS_TO_EVAL": [tracker_name],
        "SEQ_INFO": True,
    }

    print(f"Evaluando tracker: {tracker_name} ...")
    
    trackeval.run_evaluation(eval_config)


if __name__ == "__main__":
    trackers = ["tracker2_1", "tracker2_2"]  # Nombres de tus trackers
    for tracker in trackers:
        evaluate_tracker(tracker)
