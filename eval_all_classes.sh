#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

cfgs=("01Gorilla.txt" "02Unicorn.txt" "03Mallard.txt" "04Turtle.txt" "05Whale.txt" "06Bird.txt" "07Owl.txt" "08Sabertooth.txt" "09Swan.txt" "10Sheep.txt" "11Pig.txt" "12Zalika.txt" "13Pheonix.txt" "14Elephant.txt" "15Parrot.txt" "16Cat.txt" "17Scorpion.txt" "18Obesobeso.txt" "19Bear.txt" "20Puppy.txt")
clses=("01Gorilla" "02Unicorn" "03Mallard" "04Turtle" "05Whale" "06Bird" "07Owl" "08Sabertooth" "09Swan" "10Sheep" "11Pig" "12Zalika" "13Pheonix" "14Elephant" "15Parrot" "16Cat" "17Scorpion" "18Obesobeso" "19Bear" "20Puppy")
path="configs/LEGO-3D/"

len=${#cfgs[@]}

for ((i=0; i<$len; i++)); do
    python eval.py --config "$path${cfgs[i]}" --class_name "${clses[i]}"

done