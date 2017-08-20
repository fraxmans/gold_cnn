#!/bin/bash

rm data/*_small.csv
head -n $1 data/XAUUSD.csv > data/XAUUSD_small.csv
head -n $1 data/EURUSD.csv > data/EURUSD_small.csv

