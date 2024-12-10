# python train_window_from_spec.py --station_number 225 --model transformer --epochs 25 &&
# python train_window_from_spec.py --station_number 243 --model transformer --epochs 25 &&
# python train_window_from_spec.py --station_number 225 --model lstm --epochs 100 &&
# python train_window_from_spec.py --station_number 243 --model lstm --epochs 100 &&

# python train_window_from_spec.py --station_number 225 --model lstm --num_layers 4 --epochs 25 &&
# python train_window_from_spec.py --station_number 243 --model lstm --num_layers 4 --epochs 25 &&

# python train_window_from_spec.py --station_number 225 --model lstm --num_layers 6 --epochs 25 &&
# python train_window_from_spec.py --station_number 243 --model lstm --num_layers 6 --epochs 25

# python train_window_from_spec.py --station_number 225 --model enhanced_transformer --epochs 25 &&
# python train_window_from_spec.py --station_number 243 --model enhanced_transformer --epochs 25

# python train_window_from_spec.py --station_number 225 --model conv_lstm --epochs 25 &&
# python train_window_from_spec.py --station_number 243 --model conv_lstm --epochs 25

python train_window_from_spec.py --station_number 225 --model enhanced_lstm --epochs 25 &&
python train_window_from_spec.py --station_number 243 --model enhanced_lstm --epochs 25
