# Bed-monitoring

## Requirement
- Python
- Posture RSSI data

## Setup
```
$ git clone https://github.com/kenkn/bed-monitoring.git
$ cd bed-monitoring
$ pip install -r requirements.txt
```

## Execute
```
$ python clustering.py <clustering method(svc or sgd or kneigh)> <count of train(int:1~{<testers_count>-1})>
```
