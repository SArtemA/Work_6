import json
import gdown
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def to_json(filename, data):
    #filename = filename.split('.')[0]
    filename += '.json'
    print('result in:',filename)
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2, default=str)

def get_files_from_drive():

    def drive_get(id_of_file):
        url = f'https://drive.google.com/uc?id={id_of_file}'
        gdown.download(url)

    drive_ids = ['1o21DDYh4C75nWTde9snZ_h5zTEU7Y2w8',
                 '1gb3uE81z21e80QS57g3wBuHnZUi-2VHe',
                 '10wCPtuoafSuhtCBV61W5h4nu_WZmjs5-',
                 '1Xi2p7ZvJDa6dcH2TmN2_fZPHqGepVjw9',
                 '1agmAd_U-u6zVFDjnqzY-wT6qro-y0fGc',
                 '1tv_FP2GQfmV2FzJ-bkO3XirVCy0kXXtB' #мой файл
                 ]

    for drive_id in drive_ids:
        drive_get(drive_id)

def file_read(file_name, chunk_read = False, chunks_size = 10000):
    file_extension = file_name.split(".")
    if not chunk_read:
        if 'csv' not in file_extension:
            print('if if')
            return next(pd.read_csv(file_name,
                                    compression=file_extension[-1]))
        elif file_extension[1] == "csv" and len(file_extension) == 2:
            print('if elif 1')
            return pd.read_csv(file_name)
        elif file_extension[-1] == 'gz':
            print('if elif 2')
            return next(pd.read_csv(file_name,
                                    compression='gzip',
                                    chunksize=chunks_size))
        else:
            print('if else')
            return next(pd.read_csv(file_name,
                                    compression=file_extension[-1]))
    elif chunk_read:
        if 'csv' not in file_extension:
            print('elif if')
            return next(pd.read_csv(file_name,
                                    compression=file_extension[-1],
                                    chunksize=chunks_size))
        elif file_extension[1] == "csv" and len(file_extension) == 2:
            print('elif elif 1')
            return pd.read_csv(file_name,
                               chunksize=chunks_size)
        elif file_extension[-1] == 'gz':
            print('elif elif 2')
            return next(pd.read_csv(file_name,
                                    compression='gzip',
                                    chunksize=chunks_size))
        else:
            print('elif else')
            return next(pd.read_csv(file_name,
                                    compression=file_extension[-1],
                                    chunksize=chunks_size))


def get_memory_stat_by_column(df,filename):

    mem_usage_stat = df.memory_usage(deep=True)
    total_mem_usage = mem_usage_stat.sum()
    print(f"file in memory size = {total_mem_usage // 1024:10} Kb")
    column_stat = list()
    for key in df.dtypes.keys():
        column_stat.append({
                "column_name" : key,
                "memory_abs" : mem_usage_stat[key] // 1024,
                'memory_per' : round(mem_usage_stat[key] / total_mem_usage * 100, 4),
                'dtype' : df.dtypes[key]
            })

    column_stat.sort(key = lambda x: x['memory_abs'], reverse = True)

    namus = filename.split('.')[0]
    to_json(f'{namus}_memory_stat_by_column', column_stat)

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return '{:03.2f} MB'.format(usage_mb)

def opt_obj(df):
    converted_obj = pd.DataFrame()
    dataset_obj = df.select_dtypes(include=['object']).copy()

    for col in dataset_obj.columns:
        num_unique_values = len(dataset_obj[col].unique())
        num_total_values = len(dataset_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:, col] = dataset_obj[col].astype('category')
        else:
            converted_obj.loc[:, col] = dataset_obj[col]

    print(mem_usage(dataset_obj))
    print(mem_usage(converted_obj))
    return converted_obj

def opt_int(df):

    dataset_int = df.select_dtypes(include=['int'])
    converted_int = dataset_int.apply(pd.to_numeric, downcast='unsigned')

    print(mem_usage(dataset_int))
    print(mem_usage(converted_int))

    compare_ints = pd.concat([dataset_int.dtypes, converted_int.dtypes], axis=1)
    compare_ints.columns = ['before','after']
    compare_ints.apply(pd.Series.value_counts)

    print(compare_ints)
    return converted_int

def opt_float(df):

    dataset_float = df.select_dtypes(include=['float'])
    converted_float = dataset_float.apply(pd.to_numeric, downcast='float')

    print(mem_usage(dataset_float))
    print(mem_usage(converted_float))

    compare_float = pd.concat([dataset_float.dtypes, converted_float.dtypes], axis=1)
    compare_float.columns = ['before','after']
    compare_float.apply(pd.Series.value_counts)

    print(compare_float)
    return converted_float

def dataset_optimization(dataset):
    optimized_dataset = dataset.copy()
    converted_obj   = opt_obj(dataset)
    converted_int   = opt_int(dataset)
    converted_float = opt_float(dataset)

    optimized_dataset[converted_obj.columns]   = converted_obj
    optimized_dataset[converted_int.columns]   = converted_int
    optimized_dataset[converted_float.columns] = converted_float

    print(mem_usage(dataset))
    print(mem_usage(optimized_dataset))
    return optimized_dataset

def read_types(file_name):
    dtypes = dict()
    with open(file_name, mode='r') as file:
        dtypes = json.load(file)
    for key in dtypes.keys():
        if dtypes[key] == 'category':
            dtypes[key] = pd.CategoricalDtype
        else:
            dtypes[key] = np.dtype(dtypes[key])
    return dtypes

# тут графики
def plotting(file_name, optimized_dataset,dataset_columns,sd_name, chunks_size=100_000, range_start = 0, range_end = 10):

    # функции рисования графиков
    def graph_1(file_name, newest_dataset, sd_name):
        plt.figure(figsize=(60, 10))
        sort_dow = newest_dataset[sd_name].sort_index()

        plot = sort_dow.hist()
        plot.get_figure().savefig(f'{file_name}_graph_1.png')

    def graph_2(file_name, newest_dataset, sd_name):
        plt.figure(figsize=(30, 30))
        sort_dow = newest_dataset[sd_name].value_counts()

        plot = sort_dow.plot(kind = 'pie', title = f'{sd_name} scores')
        plot.get_figure().savefig(f'{file_name}_graph_2.png')

    def graph_3(file_name, newest_dataset, sd_name_1, sd_name_2):
        plt.figure(figsize=(60, 10))
        sort_dow = newest_dataset[sd_name_1].value_counts()
        plot = sort_dow.plot()
        plot.get_figure().savefig(f'{file_name}_graph_3.png')

    def graph_4(file_name, newest_dataset, sd_name):
        plt.figure(figsize=(60, 10))
        sort_dow = newest_dataset[sd_name].value_counts()
        plot = sort_dow.plot(kind='bar', title='sd_name')
        plot.get_figure().savefig(f'{file_name}_graph_4.png')

    def graph_5(file_name, newest_dataset, sd_name_1, sd_name_2):
        plt.figure(figsize=(60, 10))
        sort_dow = newest_dataset[sd_name]
        plot = sort_dow.groupby(sd_name_1)[sd_name_2].mean().plot()
        plot.get_figure().savefig(f'{file_name}_graph_5.png')

    column_names = [dataset_columns[i] for i in list(range(range_start,range_end))]
    need_column = dict()
    opt_dtypes = optimized_dataset.dtypes

    for column_name in column_names:
        need_column[column_name] = opt_dtypes[column_name]
        print(f'{column_name}:{opt_dtypes[column_name]}')

    with open(f'{file_name}_dtypes.json',mode='w') as file:
        dtype_json = need_column.copy()
        for key in dtype_json.keys():
            dtype_json[key] = str(dtype_json[key])
        json.dump(dtype_json,file)

    has_header = True
    for chunk in pd.read_csv(file_name,
                             usecols=lambda x: x in column_names,
                             dtype=need_column,
                             chunksize=chunks_size):
        print('chunk', mem_usage(chunk))
        chunk.to_csv(f"{file_name}_df.csv",mode='a',header = has_header)
        has_header = False

    need_dtypes = read_types(f"{file_name}_dtypes.json")
    newest_dataset = pd.read_csv(f"{file_name}_df.csv",
                                 usecols=lambda x: x in column_names,
                                 dtype=need_dtypes)
    newest_dataset.info(memory_usage='deep')

    # вызов функций рисования графиков
    graph_1(file_name, newest_dataset,sd_name[0])
    graph_2(file_name, newest_dataset, sd_name[1])
    graph_3(file_name, newest_dataset, sd_name[2], sd_name[3])
    graph_4(file_name, newest_dataset, sd_name[4])
    graph_5(file_name, newest_dataset, sd_name[5], sd_name[6])

# главные функции для каждого файла
def first_file(file_name):
    print(file_name)
    file_size = os.path.getsize(file_name)
    chunk_read = False
    chunk_size = file_size // 1024
    dataset = file_read(file_name,chunk_read, chunk_size)
    print(f'file size           = {file_size // 1024:10} Kb')
    get_memory_stat_by_column(dataset, file_name)

    optimized_dataset = dataset_optimization(dataset)
    #read_and_optimized = pd.read_csv(file_name,usecols=lambda x: x in column_names,dtype=need_column)
    #print(read_and_optimized.shape)
    #print(mem_usage(read_and_optimized))
    for_graphs = ['day_of_week', 'day_of_week', 'v_score', 'v_name', 'v_name','h_name', 'v_score']

    plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, 100_000)

def second_file(file_name):
    print(file_name)
    file_size = os.path.getsize(file_name)
    print(f'file size           = {file_size // 1024:10} Kb')

    chunk_read = True
    chunk_size = 1000_000
    dataset = file_read(file_name, chunk_read, chunk_size)

    new_data_name = file_name.split('.')[0]
    dataset.to_csv(f"{new_data_name}_new.csv", mode='a')
    file_name = f"{new_data_name}_new.csv"

    print(f'size of new file    = {chunk_size // 1024:10} Kb')

    get_memory_stat_by_column(dataset, file_name)

    optimized_dataset = dataset_optimization(dataset)

    # read_and_optimized = pd.read_csv(file_name,usecols=lambda x: x in column_names,dtype=need_column)
    # print(read_and_optimized.shape)
    # print(mem_usage(read_and_optimized))
    range_ln = 20
    for_graphs = ['color', 'color', 'askPrice', 'interiorColor', 'interiorColor', 'stockNum', 'askPrice']
    plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, chunk_size//10, 0, 15)

def third_file(file_name):
    print(file_name)
    file_size = os.path.getsize(file_name)
    print(f'file size           = {file_size // 1024:10} Kb')

    chunk_read = False
    chunk_size = file_size // 1024
    dataset = file_read(file_name, chunk_read, chunk_size)

    #new_data_name = file_name.split('.')[0]
    #dataset.to_csv(f"{new_data_name}_new.csv", mode='a')
    #file_name = f"{new_data_name}_new.csv"

    print(f'size of new file    = {chunk_size // 1024:10} Kb')

    get_memory_stat_by_column(dataset, file_name)

    optimized_dataset = dataset_optimization(dataset)

    # read_and_optimized = pd.read_csv(file_name,usecols=lambda x: x in column_names,dtype=need_column)
    # print(read_and_optimized.shape)
    # print(mem_usage(read_and_optimized))

    for_graphs = ['AIRLINE', 'MONTH', 'DESTINATION_AIRPORT', 'DAY', 'DAY', 'SCHEDULED_TIME', 'DEPARTURE_DELAY']
    #plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, 0, 15) # тут была ошибка ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, 100_000, 0, 15)
def fourth_file(file_name):
    print(file_name)
    file_size = os.path.getsize(file_name)
    print(f'file size           = {file_size // 1024:10} Kb')

    chunk_read = True
    chunk_size = file_size // 1024
    dataset = file_read(file_name, chunk_read, chunk_size)

    new_data_name = file_name.split('.')[0]
    dataset.to_csv(f"{new_data_name}_new.csv", mode='a')
    file_name = f"{new_data_name}_new.csv"

    print(f'size of new file    = {chunk_size // 1024:10} Kb')

    get_memory_stat_by_column(dataset, file_name)

    optimized_dataset = dataset_optimization(dataset)

    # read_and_optimized = pd.read_csv(file_name,usecols=lambda x: x in column_names,dtype=need_column)
    # print(read_and_optimized.shape)
    # print(mem_usage(read_and_optimized))

    for_graphs = ['employer_id', 'driver_license_types', 'salary_from', 'employer_id', 'employer_industries', 'type_name', 'salary_from']
    plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, 100_000, 15, 30)

def fifth_file(file_name):
    print(file_name)
    file_size = os.path.getsize(file_name)
    print(f'file size           = {file_size // 1024:10} Kb')

    chunk_read = True
    chunk_size = file_size // 1024
    dataset = file_read(file_name, chunk_read, chunk_size)

    new_data_name = file_name.split('.')[0]
    dataset.to_csv(f"{new_data_name}_new.csv", mode='a')
    file_name = f"{new_data_name}_new.csv"

    print(f'size of new file    = {chunk_size // 1024:10} Kb')

    get_memory_stat_by_column(dataset, file_name)

    optimized_dataset = dataset_optimization(dataset)

    # read_and_optimized = pd.read_csv(file_name,usecols=lambda x: x in column_names,dtype=need_column)
    # print(read_and_optimized.shape)
    # print(mem_usage(read_and_optimized))

    for_graphs = ['orbit_id', 'equinox', 'pha', 'H', 'diameter_sigma',
                  'e', 'a']
    plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, 100_000, 7, 20)

def sixth_file(file_name):
    print(file_name)
    file_size = os.path.getsize(file_name)
    print(f'file size           = {file_size // 1024:10} Kb')

    chunk_read = False
    chunk_size = file_size // 1024
    dataset = file_read(file_name, chunk_read, chunk_size)

    #new_data_name = file_name.split('.')[0]
    #dataset.to_csv(f"{new_data_name}_new.csv", mode='w')
    #file_name = f"{new_data_name}_new.csv"

    print(f'size of new file    = {chunk_size // 1024:10} Kb')

    get_memory_stat_by_column(dataset, file_name)

    optimized_dataset = dataset_optimization(dataset)

    # read_and_optimized = pd.read_csv(file_name,usecols=lambda x: x in column_names,dtype=need_column)
    # print(read_and_optimized.shape)
    # print(mem_usage(read_and_optimized))

    for_graphs = ['Vict Age', 'Vict Sex', 'AREA NAME', 'Mocodes', 'Weapon Used Cd',
                  'Weapon Desc', 'Premis Cd']
    plotting(file_name, optimized_dataset, [key for key in dataset.dtypes.keys()], for_graphs, 100_000, 3, 22)



#скачивает файлы с диска
get_files_from_drive()
pd.set_option("display.max_rows",20,"display.max_columns",60)
#обработка 1 файла
first_file("[1]game_logs.csv")
#обработка 2 файла
second_file('[2]automotive.csv.zip')
#обработка 3 файла
third_file('[3]flights.csv')
#обработка 4 файла
fourth_file('[4]vacancies.csv.gz')
#обработка 5 файла
fifth_file('[5]asteroid.zip')
#обработка 6 файла
sixth_file('Crime_Data_from_2020_to_Present.csv')





print('end')

