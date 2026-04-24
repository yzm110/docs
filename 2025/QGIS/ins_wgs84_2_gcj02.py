import pandas as pd
import math
import os
import argparse

# 定义GCJ-02转换函数
def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def wgs84_to_gcj02(lon, lat):
    if out_of_china(lon, lat):
        return lon, lat
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglon = lon + dlon
    return mglon, mglat

def out_of_china(lon, lat):
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)

a = 6378245.0
ee = 0.00669342162296594323

# 获取命令行参数
parser = argparse.ArgumentParser(description='Convert WGS84 coordinates to GCJ-02 coordinates in a CSV file.')
parser.add_argument('input_csv', help='The input CSV file path')
parser.add_argument('output_csv', help='The output CSV file path')

args = parser.parse_args()

# 读取CSV文件
df = pd.read_csv(args.input_csv)

# 定义转换公式
def convert_latitude(lat):
    return lat * 1e-7 - 180

def convert_longitude(lon):
    return lon * 1e-7 - 180

# 应用转换公式
df['Latitude'] = df['msg.OSM2_Bus_INSinf.INSLatitude'].apply(convert_latitude)
df['Longitude'] = df['msg.OSM2_Bus_INSinf.INSLongitude'].apply(convert_longitude)

# 转换为GCJ-02坐标
df[['Longitude', 'Latitude']] = df.apply(lambda row: wgs84_to_gcj02(row['Longitude'], row['Latitude']), axis=1, result_type='expand')

# 保存新的CSV文件
df.to_csv(args.output_csv, index=False)

print(f"转换完成，新的CSV文件已保存为 '{args.output_csv}'")
