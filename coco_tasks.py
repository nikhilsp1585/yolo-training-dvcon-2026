"""
coco_tasks.py
─────────────
Defines the 14 predefined task categories used in:
  "What Object Should I Use? – Task Driven Object Detection" (CVPR 2019)

Each task maps to:
  - A natural-language description (fed to DistilBERT)
  - The set of COCO class IDs that are considered task-relevant
  - A short label for plotting

COCO 80-class IDs (0-indexed, matching ultralytics YOLOv5):
  0:person, 1:bicycle, 2:car, 3:motorcycle, 4:airplane, 5:bus,
  6:train, 7:truck, 8:boat, 9:traffic light, 10:fire hydrant,
  11:stop sign, 12:parking meter, 13:bench, 14:bird, 15:cat,
  16:dog, 17:horse, 18:sheep, 19:cow, 20:elephant, 21:bear,
  22:zebra, 23:giraffe, 24:backpack, 25:umbrella, 26:handbag,
  27:tie, 28:suitcase, 29:frisbee, 30:skis, 31:snowboard,
  32:sports ball, 33:kite, 34:baseball bat, 35:baseball glove,
  36:skateboard, 37:surfboard, 38:tennis racket, 39:bottle,
  40:wine glass, 41:cup, 42:fork, 43:knife, 44:spoon, 45:bowl,
  46:banana, 47:apple, 48:sandwich, 49:orange, 50:broccoli,
  51:carrot, 52:hot dog, 53:pizza, 54:donut, 55:cake,
  56:chair, 57:couch, 58:potted plant, 59:bed, 60:dining table,
  61:toilet, 62:tv, 63:laptop, 64:mouse, 65:remote, 66:keyboard,
  67:cell phone, 68:microwave, 69:oven, 70:toaster, 71:sink,
  72:refrigerator, 73:book, 74:clock, 75:vase, 76:scissors,
  77:teddy bear, 78:hair drier, 79:scissors
"""

COCO_TASKS = {
    "T01_cooking": {
        "description": "I want to cook a meal. What objects do I need?",
        "relevant_ids": [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                         50, 51, 52, 53, 54, 55, 68, 69, 70, 71],
        "label": "Cooking"
    },
    "T02_eating": {
        "description": "I want to eat food. What objects should I use?",
        "relevant_ids": [41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                         51, 52, 53, 54, 55, 60],
        "label": "Eating"
    },
    "T03_travel": {
        "description": "I am travelling. What transportation objects are relevant?",
        "relevant_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 28],
        "label": "Travel"
    },
    "T04_sports_outdoor": {
        "description": "I want to do outdoor sports. What equipment do I need?",
        "relevant_ids": [29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        "label": "Outdoor sports"
    },
    "T05_office_work": {
        "description": "I am working in an office. What objects are useful?",
        "relevant_ids": [62, 63, 64, 65, 66, 67, 73, 74, 76],
        "label": "Office work"
    },
    "T06_pet_care": {
        "description": "I want to take care of my pet. What animals do I see?",
        "relevant_ids": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "label": "Pet care"
    },
    "T07_home_relaxing": {
        "description": "I want to relax at home. What furniture and objects are available?",
        "relevant_ids": [56, 57, 58, 59, 60, 61, 62, 65, 73, 74, 75, 77],
        "label": "Home relaxing"
    },
    "T08_personal_hygiene": {
        "description": "I need to get ready and maintain hygiene. What objects do I need?",
        "relevant_ids": [71, 72, 78],
        "label": "Personal hygiene"
    },
    "T09_drinking": {
        "description": "I want to have a drink. What objects should I use?",
        "relevant_ids": [39, 40, 41, 46, 49],
        "label": "Drinking"
    },
    "T10_safety_traffic": {
        "description": "I need to be aware of traffic safety. What objects are relevant?",
        "relevant_ids": [0, 1, 2, 3, 5, 7, 9, 10, 11, 12],
        "label": "Traffic safety"
    },
    "T11_outdoor_recreation": {
        "description": "I am going for outdoor recreation. What objects are present?",
        "relevant_ids": [13, 24, 25, 26, 29, 30, 31, 33, 36, 37],
        "label": "Outdoor recreation"
    },
    "T12_formal_dressing": {
        "description": "I need to dress formally. What clothing accessories are present?",
        "relevant_ids": [24, 25, 26, 27, 28],
        "label": "Formal dressing"
    },
    "T13_photography": {
        "description": "I want to take photos or record video. What devices are available?",
        "relevant_ids": [62, 63, 67, 74],
        "label": "Photography"
    },
    "T14_reading_learning": {
        "description": "I want to read and learn something. What objects are useful?",
        "relevant_ids": [62, 63, 66, 73],
        "label": "Reading/learning"
    },
}

# Convenience: all 14 task keys in order
TASK_KEYS = list(COCO_TASKS.keys())

# COCO class name list (ultralytics order, 80 classes)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","scissors"
]
