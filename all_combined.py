import warnings
import traceback
from datetime import datetime
from functools import partial
import numpy
import numpy as np
import pandas as pd
import multiprocessing
import requests
import os


warnings.filterwarnings('error', category=RuntimeWarning)

master_df = pd.DataFrame()
fix_val = 1000


# writing all other files except daily summary logic
# def excel_write(excel_name, sheet_name, dataframe):
#     save_location = 'C:/Users/user/Desktop/summary_files/'
#     excel_file_path = os.path.join(save_location, f"{excel_name}.xlsx")
#
#     if os.path.exists(excel_file_path):
#         with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
#             try:
#                 writer.book[sheet_name]
#                 writer.book.remove(writer.sheets[sheet_name])
#             except KeyError:
#                 pass
#
#             dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
#     else:
#         with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
#             dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

# def excel_write2(excel_name, sheet_name, dataframe):
#     save_location = 'C:/Users/user/Desktop/summary_files/'
#     excel_file_path = os.path.join(save_location, f"{excel_name}.xlsx")
#
#     if os.path.exists(excel_file_path):
#         with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
#             try:
#                 writer.book[sheet_name]
#                 writer.book.remove(writer.sheets[sheet_name])
#             except KeyError:
#                 pass
#
#             dataframe.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
#     else:
#         with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
#             dataframe.to_excel(writer, sheet_name=sheet_name, index=False, header=False)



# writing excel and creating of directory logic
def write_to_excel(df, name, dir_name):
    dir_name = str(dir_name)
    dir_path = f'C:/Users/hetnk/Desktop/summary_files/{dir_name}/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, f'{name}.xlsx')

    # Check if the file exists
    if os.path.exists(file_path):
        existing_data = pd.read_excel(file_path)
        combined_data = pd.concat([existing_data, df], ignore_index=True)
        combined_data.to_excel(file_path, index=False)
    else:
        df.to_excel(file_path, index=False)


def daily_calc(df):
    val = []

    df.at[0, 'symbol'] = df.at[0, 'Symbol']
    df.at[1, 'symbol'] = df.at[1, 'Symbol']
    df.at[0, 'symbol'] = pd.to_datetime(df.at[0, 'symbol'])
    df.at[0, 'symbol'] = pd.to_datetime(df.at[0, 'symbol'])
    day1 = df.at[0, 'symbol'].weekday()
    if day1 == 4:
        df.at[2, 'symbol'] = df.at[0, 'symbol'] + pd.Timedelta(days=3)
    else:
        df.at[2, 'symbol'] = df.at[0, 'symbol'] + pd.Timedelta(days=1)

    df['High'].replace(' ', np.nan)
    df['High'].dropna()

    first_val = df.loc[0, 'Close']
    for index, col in df.iterrows():
        next_close = col['Close']
        msf_val = round((next_close - first_val) * 0.33 + first_val, 2)
        val.append(msf_val)
        first_val = msf_val
    df['MSF'] = val

    condition_upmove = (
        (df['Close'].shift(1) < df['High']) &
        (df['Close'].shift(1) < df['Close']) &
        (df['High'].shift(1) < df['High'])
    )

    condition_downmove = (
        (df['Close'].shift(1) > df['Low']) &
        (df['Close'].shift(1) > df['Close']) &
        (df['Low'].shift(1) > df['Low'])
    )

    df['Move'] = ''
    df.loc[condition_upmove, 'Move'] = 'UPMOVE'
    df.loc[condition_downmove, 'Move'] = 'DOWNMOVE'


    df['MSF_Colour'] = ''

    condition_msf_col_3 = (
            (df['MSF'] < df['MSF'].shift(1)) &
            (df['MSF'] > df['MSF'].shift(2))
    )

    condition_msf_col_4 = (
            (df['MSF'] > df['MSF'].shift(1)) &
            (df['MSF'] < df['MSF'].shift(2))
    )

    condition_msf_col_5 = (
            (df['MSF'] > df['MSF'].shift(1)) &
            (df['MSF'] > df['MSF'].shift(2))
    )

    condition_msf_col_6 = (
            (df['MSF'] < df['MSF'].shift(1)) &
            (df['MSF'] < df['MSF'].shift(2))
    )



    df.loc[(df.index >= 3) & condition_msf_col_3, 'MSF_Colour'] = 'z.green'
    df.loc[(df.index >= 3) & condition_msf_col_4, 'MSF_Colour'] = 'z.red'
    df.loc[(df.index >= 3) & condition_msf_col_5, 'MSF_Colour'] = 'green'
    df.loc[(df.index >= 3) & condition_msf_col_6, 'MSF_Colour'] = 'red'



    condition_msf_col_1 = (
            (df['MSF'] > df['MSF'].shift(1)) &
            (df['MSF'] > df['MSF'].shift(2)) &
            ((df['MSF_Colour'].shift(1) == 'red') | (df['MSF_Colour'].shift(1) == 'z.red'))

    )

    condition_msf_col_2 = (
            (df['MSF'] < df['MSF'].shift(1)) &
            (df['MSF'] < df['MSF'].shift(2)) &
            ((df['MSF_Colour'].shift(1) == 'green') | (df['MSF_Colour'].shift(1) == 'z.green'))
    )

    df.loc[condition_msf_col_1, 'MSF_Colour'] = '1st sunrise'
    df.loc[condition_msf_col_2, 'MSF_Colour'] = '1st sunset'


    df['Range'] = df['High'] - df['Low']
    df['Range'] = np.nan_to_num(df['Range'], nan=0)
    df['JGD'] = np.ceil((df['High'] - (df['Range'] * 0.382)) / 0.1) * 0.1
    df['JWD'] = np.floor((df['Low'] + (df['Range'] * 0.382)) / 0.1) * 0.1
    df['JGD'] = np.nan_to_num(df['JGD'], nan=0)
    df['JWD'] = np.nan_to_num(df['JWD'], nan=0)



    condition_1_1 = (
        (df['JWD'] > df['JWD'].shift(1).fillna(0))
    )

    condition_2_2 = (
        (df['JGD'] < df['JWD'].shift(1).fillna(0))
    )

    condition_3_3 = ~condition_1_1 & ~ condition_2_2

    df['D_patt'] = ''
    df.loc[condition_1_1, 'D_patt'] = '2+2'
    df.loc[condition_2_2, 'D_patt'] = '3+1'
    df.loc[condition_3_3, 'D_patt'] = '2+1'


    df.loc[0, 'E6'] = df.loc[0, 'Close']
    i = 1
    while i <= len(df):
        df.loc[i:, 'E6'] = round(df['E6'].shift(1) * (1 - 0.2857) + (df['Close'] * 0.2857), 2)
        i += 1


    df.loc[0, 'E30'] = df.loc[0, 'Close']
    i = 1
    while i <= len(df):
        df.loc[i:, 'E30'] = round(df['E30'].shift(1) * (1 - 0.0645) + (df['Close'] * 0.0645), 2)
        i += 1



    df.loc[0, 'E65'] = df.loc[0, 'Close']
    i = 1
    while i <= len(df):
        df.loc[i:, 'E65'] = round(df['E65'].shift(1) * (1 - 0.0303) + (df['Close'] * 0.0303), 2)
        i += 1



    df['CP'] = round((df['Close'] + df['E6']) / 2, 2)
    df['H.MCP'] = round((df['Low'] + df['E6'].shift(1))/2, 2)
    df['L.MCP'] = round((df['High'] + df['E6'].shift(1))/2, 2)


    condition_i1 = (
        (df['High'] < df['High'].shift(1)) &
        (df['Low'] > df['Low'].shift(1))
    )

    condition_i2 = (
        (df['High'] > df['High'].shift(1)) &
        (df['Low'] < df['Low'].shift(1))
    )

    df['I'] = ''
    df.loc[condition_i1, 'I'] = 'INSIDE DAY'
    df.loc[condition_i2, 'I'] = ' '


    df.loc[1:, 'Z'] = df['Close']


    df.at[0, 'AA'] = 775
    i = 1
    while i <= len(df):
        df.loc[i:, 'AA'] = round((df['Close'] - df['AA'].shift(1)) * 0.33 + df['AA'].shift(1), 2)
        i += 1


    range_val = df['Range']
    close_val = df['Close']
    df.loc[1:, 'AD'] = round((range_val * 0.382) + close_val, 2)
    df.loc[1:, 'AE'] = round(close_val - (range_val * 0.382), 2)


    df['AH'] = 0
    df['AI'] = 0
    df['AK'] = ''


    for i in range(1, len(df)):
        try:
            if df.at[i, 'AA'] > df.at[i-1, 'AH'] and df.at[i, 'AA'] > df.at[i - 1, 'AI']:
                df.at[i, 'AF'] = 'GREEN'
            elif df.at[i, 'AA'] < df.at[i-1, 'AH'] and df.at[i, 'AA'] < df.at[i - 1, 'AI']:
                df.at[i, 'AF'] = 'RED'
            elif df.at[i - 1, 'AH'] > df.at[i - 1, 'AI'] or df.at[i - 1, 'AH'] < df.at[i - 1, 'AI'] or df.at[i - 1, 'AH'] == df.at[i - 1, 'AI']:
                df.at[i, 'AF'] = 'ZIGZAG'

            df['AC'] = df['AF']

            if df.at[i, 'AF'] == 'GREEN':
                df.at[i, 'AG'] = 'GREEN'
            elif df.at[i, 'AF'] == 'RED':
                df.at[i, 'AG'] = 'RED'
            elif df.at[i, 'AF'] == 'ZIGZAG' and df.at[i-1, 'AF'] == 'GREEN':
                df.at[i, 'AG'] = 'GREEN ZIGZAG'
            elif df.at[i, 'AF'] == 'ZIGZAG' and df.at[i-1, 'AF'] == 'RED':
                df.at[i, 'AG'] = 'RED ZIGZAG'
            elif df.at[i, 'AF'] == 'ZIGZAG' and df.at[i-1, 'AF'] == 'ZIGZAG':
                df.at[i, 'AG'] = df.at[i-1, 'AG']

            if df.at[i, 'AG'] == 'GREEN ZIGZAG' and df.at[i - 1, 'AG'] != 'GREEN ZIGZAG' and df.at[i - 1, 'AG'] != 'RED ZIGZAG':
                df.at[i, 'AH'] = df.at[i - 1, 'AA']
            elif df.at[i, 'AF'] == 'ZIGZAG':
                df.at[i, 'AH'] = df.at[i - 1, 'AH']
            elif df.at[i, 'AF'] == 'GREEN':
                df.at[i, 'AH'] = df.at[i, 'AA']
            elif df.at[i, 'AF'] == 'RED':
                df.at[i, 'AH'] = df.at[i, 'AA']
            #
            if df.at[i, 'AG'] == 'GREEN ZIGZAG' and df.at[i - 1, 'AG'] != 'GREEN ZIGZAG' and df.at[i - 1, 'AG'] != 'RED ZIGZAG':
                df.at[i, 'AI'] = df.at[i - 2, 'AA']
            elif df.at[i, 'AF'] == 'ZIGZAG':
                df.at[i, 'AI'] = df.at[i - 1, 'AI']
            elif df.at[i, 'AF'] == 'GREEN':
                df.at[i, 'AI'] = df.at[i - 1, 'AA']
            elif df.at[i, 'AF'] == 'RED':
                df.at[i, 'AI'] = df.at[i - 1, 'AA']

            if df.at[i - 1, 'AG'] == 'GREEN' and df.at[i, 'AG'] == 'RED':
                df.at[i, 'AJ'] = 'SUNSET'
            elif df.at[i - 1, 'AG'] == 'RED' and df.at[i, 'AG'] == 'GREEN':
                df.at[i, 'AJ'] = 'SUNRISE'
            elif df.at[i - 1, 'AG'] == 'GREEN ZIGZAG' and df.at[i, 'AG'] == 'RED':
                df.at[i, 'AJ'] = 'SUNSET'
            elif df.at[i - 1, 'AG'] == 'RED ZIGZAG' and df.at[i, 'AG'] == 'GREEN':
                df.at[i, 'AJ'] = 'SUNRISE'
            else:
                df.at[i, 'AJ'] = df.at[i, 'AG']



            if df.at[i - 1, 'AK'] == 'SUNSET' and df.at[i, 'AG'] == 'GREEN':
                df.at[i, 'AK'] = 'GREEN'
            elif df.at[i - 1, 'AK'] == 'SUNRISE' and df.at[i, 'AG'] == 'RED':
                df.at[i, 'AK'] = 'RED'
            elif df.at[i, 'AJ'] == 'SUNSET':
                df.at[i, 'AK'] = 'SUNSET'
            elif df.at[i, 'AJ'] == 'SUNRISE':
                df.at[i, 'AK'] = 'SUNRISE'
            elif df.at[i - 1, 'AK'] == 'SUNSET' and df.at[i, 'Z'] > df.at[i - 1, 'AE']:
                df.at[i, 'AK'] = 'SUNSET'
            elif df.at[i - 1, 'AK'] == 'SUNSET' and df.at[i, 'Z'] < df.at[i - 1, 'AE']:
                df.at[i, 'AK'] = 'RED'
            elif df.at[i - 1, 'AK'] == 'SUNRISE' and df.at[i, 'Z'] < df.at[i - 1, 'AD']:
                df.at[i, 'AK'] = 'SUNRISE'
            elif df.at[i - 1, 'AK'] == 'SUNRISE' and df.at[i, 'Z'] > df.at[i - 1, 'AD']:
                df.at[i, 'AK'] = 'GREEN'
            elif df.at[i - 1, 'AK'] == 'RED' or df.at[i - 1, 'AK'] == 'GREEN' or df.at[i - 1, 'AK'] == 'RED ZIGZAG' or df.at[i - 1, 'AK'] == 'GREEN ZIGZAG':
                df.at[i, 'AK'] = df.at[i, 'AJ']
            else:
                ''


        except Exception as exc:
            print(traceback.format_exc())        # shows where the traceback occured, why it occured, when it occured and at which line did it occur


    df.at[0, 'symbol'] = df.at[0, 'symbol'].strftime('%d-%b-%Y')
    df.at[2, 'symbol'] = df.at[2, 'symbol'].strftime('%d-%b-%Y')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%d-%b-%Y')
    selected_cols = ['symbol', 'Date', 'High', 'Low', 'Close', 'Move', 'MSF', 'MSF_Colour', 'JGD', 'JWD', 'D_patt', 'Range', 'E6', 'E30', 'E65', 'CP',
                     'H.MCP', 'L.MCP', 'I', 'Z', 'AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK']

    new_df = df[selected_cols].copy()
    name1 = df.at[1, 'symbol']
    # excel_write(name1, 'Daily', new_df)
    return new_df



def weekly_calc(df):
    df_weekly = pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'])
    df_weekly['to_date'] = df['Date'] - pd.to_timedelta(df['Date'].dt.weekday, unit='D')
    df_weekly['from_date'] = df_weekly['to_date'] + pd.Timedelta(days=6)
    df_weekly = df_weekly.drop_duplicates(subset=['to_date', 'from_date']).reset_index(drop=True)


    df['Date'] = pd.to_datetime(df['Date'])


    def calculate_values(row):
        to_date = row['to_date']
        from_date = row['from_date']
        df_condition = df[(df['Date'] >= to_date) & (df['Date'] <= from_date)]
        high_val = df_condition['High'].max()
        low_val = df_condition['Low'].min()
        close_val = df_condition['Close'].iloc[-1] if not df_condition.empty else None
        return pd.Series({'High': high_val, 'Low': low_val, 'Close': close_val})

    result = df_weekly.apply(calculate_values, axis=1)
    df_weekly = pd.concat([df_weekly, result], axis=1)


    condition_move1 = (
        (df_weekly['Close'].shift(1) > df_weekly['Low']) &
        (df_weekly['Close'].shift(1) > df_weekly['Close']) &
        (df_weekly['Low'].shift(1) > df_weekly['Low'])
    )
    condition_move2 = (
        (df_weekly['Close'].shift(1) < df_weekly['High']) &
        (df_weekly['Close'].shift(1) < df_weekly['Close']) &
        (df_weekly['High'].shift(1) < df_weekly['High'])
    )

    df_weekly['MOVE'] = ''
    df_weekly.loc[condition_move1, 'MOVE'] = 'DOWNMOVE'
    df_weekly.loc[condition_move2, 'MOVE'] = 'UPMOVE'

    first_val = df_weekly.at[0, 'Close']
    msf_val_list = []
    for x, y in df_weekly.iterrows():
        close_val = y['Close']
        msf_val = round((close_val - first_val) * 0.33 + first_val, 2)
        first_val = msf_val
        msf_val_list.append(msf_val)
    df_weekly['MSF'] = msf_val_list



    df_weekly['MSF_COLOR'] = ''
    df_weekly.loc[2, 'MSF_COLOR'] = 'Red'
    if 'MSF_COLOR' in df_weekly.columns:
        df_weekly.at[2, 'MSF_COLOR'] = df_weekly.at[2, 'MSF_COLOR'].lower()
    for i in range(3, len(df_weekly)):
        if (df_weekly.at[i, 'MSF'] > df_weekly.at[i - 1, 'MSF'] and df_weekly.at[i, 'MSF'] > df_weekly.at[
            i - 2, 'MSF']) and (
                df_weekly.at[i - 1, 'MSF_COLOR'] == 'red' or df_weekly.at[i - 1, 'MSF_COLOR'] == 'z.red'):
            df_weekly.at[i, 'MSF_COLOR'] = '1st sunrise'
        elif (df_weekly.at[i, 'MSF'] < df_weekly.at[i - 1, 'MSF'] and df_weekly.at[i, 'MSF'] < df_weekly.at[
            i - 2, 'MSF']) and (
                df_weekly.at[i - 1, 'MSF_COLOR'] == 'green' or df_weekly.at[i - 1, 'MSF_COLOR'] == 'z.green' or
                df_weekly.at[i - 1, 'MSF_COLOR'] == 'GREEN'
                or df_weekly.at[i - 1, 'MSF_COLOR'] == 'Green' or df_weekly.at[i - 1, 'MSF_COLOR'] == 'Z.Green'):
            df_weekly.at[i, 'MSF_COLOR'] = '1st sunset'
        elif df_weekly.at[i, 'MSF'] < df_weekly.at[i - 1, 'MSF'] and df_weekly.at[i, 'MSF'] > df_weekly.at[
            i - 2, 'MSF']:
            df_weekly.at[i, 'MSF_COLOR'] = 'z.green'
        elif df_weekly.at[i, 'MSF'] > df_weekly.at[i - 1, 'MSF'] and df_weekly.at[i, 'MSF'] < df_weekly.at[
            i - 2, 'MSF']:
            df_weekly.at[i, 'MSF_COLOR'] = 'z.red'
        elif df_weekly.at[i, 'MSF'] > df_weekly.at[i - 1, 'MSF'] and df_weekly.at[i, 'MSF'] > df_weekly.at[
            i - 2, 'MSF']:
            df_weekly.at[i, 'MSF_COLOR'] = 'green'
        elif df_weekly.at[i, 'MSF'] < df_weekly.at[i - 1, 'MSF'] and df_weekly.at[i, 'MSF'] < df_weekly.at[
            i - 2, 'MSF']:
            df_weekly.at[i, 'MSF_COLOR'] = 'red'
        else:
            ''


    df_weekly['High'] = df_weekly['High'].fillna(0)
    df_weekly['Low'] = df_weekly['Low'].fillna(0)
    df_weekly['RANGE'] = df_weekly['High'] - df_weekly['Low']
    df_weekly['JGD'] = np.ceil((df_weekly['High'] - (df_weekly['RANGE'] * 0.382)) / 0.1) * 0.1
    df_weekly['JWD'] = np.floor((df_weekly['Low'] + (df_weekly['RANGE'] * 0.382)) / 0.1) * 0.1
    df_weekly['RANGE'] = df_weekly['RANGE'].fillna(0)
    df_weekly['JGD'] = df_weekly['JGD'].fillna(0)
    df_weekly['JWD'] = df_weekly['JWD'].fillna(0)

    condition_d_patt1 = (
        (df_weekly['JWD'] > df_weekly['JWD'].shift(1).fillna(0))
    )
    condition_d_patt2 = (
        (df_weekly['JGD'] < df_weekly['JWD'].shift(1).fillna(0))
    )
    condition_d_patt3 = ~ condition_d_patt1 & ~ condition_d_patt2

    df_weekly.loc[condition_d_patt1, 'D_PATT'] = '2+2'
    df_weekly.loc[condition_d_patt2, 'D_PATT'] = '3+1'
    df_weekly.loc[condition_d_patt3, 'D_PATT'] = '2+1'



    df_weekly['D_PATT_COLOR'] = ''
    df_weekly['D_PATT_COLOR'] = np.select(
        [
            (df_weekly['D_PATT'] == '2+2'),
            (df_weekly['D_PATT'] == '3+1'),
            (df_weekly['Close'] < df_weekly['JWD']),
            (df_weekly['Close'] > df_weekly['JGD'].shift(1)),
            (df_weekly['Close'] > df_weekly['Close'].shift(1)),
        ],
        [
            'Green',
            'Red',
            'Red',
            'Green',
            'Green',
        ],
        default='Red'  # Default color if none of the conditions are met
    )

    df_weekly.loc[0, 'D_PATT_COLOR'] = ''
    df_weekly['D_PATT_COLOR'].fillna(method='ffill', inplace=True)

    try:
        df_weekly.loc[0, 'E6'] = df_weekly.loc[0, 'Close']
        i = 1
        while i <= len(df_weekly):
            df_weekly.loc[i:, 'E6'] = round(df_weekly['E6'].shift(1) * (1 - 0.2857) + (df_weekly['Close'] * 0.2857), 2)
            i += 1


        df_weekly.loc[0, 'E30'] = df_weekly.loc[0, 'Close']
        i = 1
        while i <= len(df_weekly):
            df_weekly.loc[i:, 'E30'] = round(df_weekly['E30'].shift(1) * (1 - 0.0645) + (df_weekly['Close'] * 0.0645), 2)
            i += 1


        df_weekly.loc[0, 'E65'] = df_weekly.loc[0, 'Close']
        i = 1
        while i <= len(df_weekly):
            df_weekly.loc[i:, 'E65'] = round(df_weekly['E65'].shift(1) * (1 - 0.0377) + (df_weekly['Close'] * 0.0377), 2)
            i += 1
    except KeyError:
        pass


    df_weekly['CP'] = round((df_weekly['Close'] + df_weekly['E6']) / 2, 2)

    df_weekly['U.TGT 1'] = round(df_weekly['Close'].shift(1) + (df_weekly['RANGE'].shift(1) / 2), 2)
    df_weekly['U.TGT 2'] = round(df_weekly['Close'].shift(1) + df_weekly['RANGE'].shift(1), 2)
    df_weekly['U.TGT 3'] = round(df_weekly['Close'].shift(1) + (df_weekly['RANGE'].shift(1) * 1.5), 2)

    df_weekly['L.TGT 1'] = round(df_weekly['Close'].shift(1) - (df_weekly['RANGE'].shift(1) / 2), 2)
    df_weekly['L.TGT 2'] = round(df_weekly['Close'].shift(1) - df_weekly['RANGE'].shift(1), 2)
    df_weekly['L.TGT 3'] = round(df_weekly['Close'].shift(1) - (df_weekly['RANGE'].shift(1) * 1.5), 2)


    df_weekly['H.MCP'] = round((df_weekly['Low'] + df_weekly['E6'].shift(1)) / 2, 2)
    df_weekly['L.MCP'] = round((df_weekly['High'] + df_weekly['E6'].shift(1)) / 2, 2)

    df_weekly['to_date'] = df_weekly['to_date'].dt.strftime('%d-%b-%Y')
    df_weekly['from_date'] = df_weekly['from_date'].dt.strftime('%d-%b-%Y')

    name2 = df.at[1, 'symbol']
    # excel_write(name2, 'Weekly', df_weekly)
    return df_weekly



def monthly_calc(df1):

    df = pd.DataFrame()

    df['to'] = df1['Date'].apply(lambda x: x.replace(day=1))
    val_date = df['to'] + pd.DateOffset(months=1)
    df['from'] = val_date - pd.Timedelta(days=1)
    df = df.drop_duplicates(subset=['to']).reset_index(drop=True)

    df1['Date'] = pd.to_datetime(df1['Date'])
    df['to'] = pd.to_datetime(df['to'])
    df['from'] = pd.to_datetime(df['from'])

    def calculate_values(row):
        to_date = row['to']
        from_date = row['from']
        df_condition = df1[(df1['Date'] >= to_date) & (df1['Date'] <= from_date)]
        high_val = df_condition['High'].max()
        low_val = df_condition['Low'].min()
        close_val = df_condition['Close'].iloc[-1] if not df_condition.empty else None
        return pd.Series({'HIGH': high_val, 'LOW': low_val, 'CLOSE': close_val})

    result = df.apply(calculate_values, axis=1)
    df = pd.concat([df, result], axis=1)


    condition_move1 = (
            (df['CLOSE'].shift(1) > df['LOW']) &
            (df['CLOSE'].shift(1) > df['CLOSE']) &
            (df['LOW'].shift(1) > df['LOW'])
    )
    condition_move2 = (
            (df['CLOSE'].shift(1) < df['HIGH']) &
            (df['CLOSE'].shift(1) < df['CLOSE']) &
            (df['HIGH'].shift(1) < df['HIGH'])
    )

    df['MOVE'] = ''
    df.loc[condition_move1, 'MOVE'] = 'DOWNMOVE'
    df.loc[condition_move2, 'MOVE'] = 'UPMOVE'

    df.at[0, 'MSF'] = df.at[0, 'CLOSE']
    i = 0
    while i <= fix_val:
        df.loc[1:, 'MSF'] = round((df['CLOSE'] - df['MSF'].shift(1)) * 0.33 + df['MSF'].shift(1), 2)
        i += 1

    df.loc[0, 'MSF_COLOR'] = 'green'
    df['MSF'] = np.nan_to_num(df['MSF'], nan=0)

    large_number = 10 ** 1000

    if 'MSF_COLOR' in df.columns:
        df.at[0, 'MSF_COLOR'] = df.at[0, 'MSF_COLOR'].lower()

    for i in range(1, len(df)):
        if i <= 2:
            if df.at[i, 'MSF'] < df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] > large_number:
                df.at[i, 'MSF_COLOR'] = 'z.green'
            elif df.at[i, 'MSF'] > df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] < large_number:
                df.at[i, 'MSF_COLOR'] = 'z.red'
            elif df.at[i, 'MSF'] > df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] > large_number:
                df.at[i, 'MSF_COLOR'] = 'green'
            elif df.at[i, 'MSF'] < df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] < large_number:
                df.at[i, 'MSF_COLOR'] = 'red'
            else:
                ''
        else:
            if df.at[i, 'MSF'] < df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] > df.at[i - 2, 'MSF']:
                df.at[i, 'MSF_COLOR'] = 'z.green'
            elif df.at[i, 'MSF'] > df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] < df.at[i - 2, 'MSF']:
                df.at[i, 'MSF_COLOR'] = 'z.red'
            elif df.at[i, 'MSF'] > df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] > df.at[i - 2, 'MSF']:
                df.at[i, 'MSF_COLOR'] = 'green'
            elif df.at[i, 'MSF'] < df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] < df.at[i - 2, 'MSF']:
                df.at[i, 'MSF_COLOR'] = 'red'
            else:
                ''


    for i in range(2, len(df)):
        if (df.at[i, 'MSF'] > df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] > df.at[i - 2, 'MSF']) and (
                df.at[i - 1, 'MSF_COLOR'] == 'red' or df.at[i - 1, 'MSF_COLOR'] == 'z.red'):
            df.at[i, 'MSF_COLOR'] = '1st sunrise'
        elif (df.at[i, 'MSF'] < df.at[i - 1, 'MSF'] and df.at[i, 'MSF'] < df.at[i - 2, 'MSF']) and (
                df.at[i - 1, 'MSF_COLOR'] == 'green' or df.at[i - 1, 'MSF_COLOR'] == 'z.green'):
            df.at[i, 'MSF_COLOR'] = '1st sunset'



    df['HIGH'] = df['HIGH'].fillna(0)
    df['LOW'] = df['LOW'].fillna(0)
    df['RANGE'] = df['HIGH'] - df['LOW']
    df['JGD'] = np.ceil((df['HIGH'] - (df['RANGE'] * 0.382)) / 0.1) * 0.1
    df['JWD'] = np.floor((df['LOW'] + (df['RANGE'] * 0.382)) / 0.1) * 0.1
    df['RANGE'] = df['RANGE'].fillna(0)
    df['JGD'] = df['JGD'].fillna(0)
    df['JWD'] = df['JWD'].fillna(0)
    condition_d_patt1 = (
        (df['JWD'] > df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt2 = (
        (df['JGD'] < df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt3 = ~condition_d_patt1 & ~condition_d_patt2

    df.loc[condition_d_patt1, 'D_PATT'] = '2+2'
    df.loc[condition_d_patt2, 'D_PATT'] = '3+1'
    df.loc[condition_d_patt3, 'D_PATT'] = '2+1'


    df['D_PATT_COLOR'] = np.select(
        [
            (df['D_PATT'] == '2+2'),
            (df['D_PATT'] == '3+1'),
            (df['HIGH'] < df['JWD']),
            (df['HIGH'] > df['JGD'].shift(1)),
            (df['HIGH'] > df['HIGH'].shift(1)),
        ],
        [
            'Green',
            'Red',
            'Red',
            'Green',
            'Green',
        ],
        default='Red'  # Default color if none of the conditions are met
    )


    df.loc[0, 'D_PATT_COLOR'] = ''
    df['D_PATT_COLOR'].fillna(method='ffill', inplace=True)



    df.loc[0, 'E6'] = df.loc[0, 'CLOSE']
    i = 0
    while i <= fix_val:
        df.loc[1:, 'E6'] = round(df['E6'].shift(1) * (1 - 0.2857) + (df['CLOSE'] * 0.2857), 2)
        i += 1

    df.loc[0, 'E30'] = df.loc[0, 'CLOSE']
    i = 0
    while i <= fix_val:
        df.loc[1:, 'E30'] = round(df['E30'].shift(1) * (1 - 0.0645) + (df['CLOSE'] * 0.0645), 2)
        i += 1

    df['CP'] = round((df['CLOSE'] + df['E6']) / 2, 2)

    df['U.TGT 1'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) / 2), 2)
    df['U.TGT 2'] = round(df['CLOSE'].shift(1) + df['RANGE'].shift(1), 2)
    df['U.TGT 3'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) * 1.5), 2)

    df['L.TGT 1'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) / 2), 2)
    df['L.TGT 2'] = round(df['CLOSE'].shift(1) - df['RANGE'].shift(1), 2)
    df['L.TGT 3'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) * 1.5), 2)

    df['H.MCP'] = round((df['LOW'] + df['E6'].shift(1)) / 2, 2)
    df['L.MCP'] = round((df['HIGH'] + df['E6'].shift(1)) / 2, 2)

    df['to'] = df['to'].dt.strftime('%d-%b-%Y')
    df['from'] = df['from'].dt.strftime('%d-%b-%Y')

    select_cols = ['to', 'from', 'HIGH', 'LOW', 'CLOSE', 'MOVE', 'MSF', 'MSF_COLOR', 'JGD', 'JWD', 'D_PATT',
                   'D_PATT_COLOR', 'RANGE', 'E6', 'E30', 'CP', 'U.TGT 1', 'U.TGT 2', 'U.TGT 3', 'L.TGT 1', 'L.TGT 2', 'L.TGT 3', 'H.MCP', 'L.MCP']

    new_df2 = df[select_cols].copy()
    name3 = df1.at[1, 'symbol']
    # excel_write(name3, 'Monthly', new_df2)
    return new_df2



def quaterly_calc(df1):


    df = pd.DataFrame(index=range(100))
    df1['Date'] = pd.to_datetime(df1['Date'])


    df['FROM'] = df1['Date'].apply(lambda x: x.replace(day=1, month=1) if x.month <= 3 else
    (x.replace(day=1, month=4) if x.month > 3 and x.month <= 6 else
    (x.replace(day=1, month=7) if x.month > 6 and x.month <= 9 else
    x.replace(day=1, month=10))))


    df['TO'] = df1['Date'].apply(lambda x: x.replace(day=31, month=3) if x.month <= 3 else
    (x.replace(day=30, month=6) if x.month > 3 and x.month <= 6 else
    (x.replace(day=30, month=9) if x.month > 6 and x.month <= 9 else
    x.replace(day=31, month=12))))

    df = df.drop_duplicates(subset=['FROM', 'TO']).reset_index(drop=True)


    def calculate_values(row):
        from_date = row['FROM']
        to_date = row['TO']
        df_condition = df1[(df1['Date'] >= from_date) & (df1['Date'] <= to_date)]
        high_val = df_condition['High'].max()
        low_val = df_condition['Low'].min()
        close_val = df_condition['Close'].iloc[-1] if not df_condition.empty else None
        return pd.Series({'HIGH': high_val, 'LOW': low_val, 'CLOSE': close_val})

    result = df.apply(calculate_values, axis=1)
    df = pd.concat([df, result], axis=1)


    df['RANGE'] = df['HIGH'] - df['LOW']
    df['JGD'] = np.ceil((df['HIGH'] - (df['RANGE'] * 0.382)) / 0.1) * 0.1
    df['JWD'] = np.floor((df['LOW'] + (df['RANGE'] * 0.382)) / 0.1) * 0.1

    condition_d_patt1 = (
        (df['JWD'] > df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt2 = (
        (df['JGD'] < df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt3 = ~condition_d_patt1 & ~condition_d_patt2

    df.loc[(df.index >= 1) & condition_d_patt1, 'D_PATT'] = '2+2'
    df.loc[(df.index >= 1) & condition_d_patt2, 'D_PATT'] = '3+1'
    df.loc[(df.index >= 1) & condition_d_patt3, 'D_PATT'] = '2+1'

    df.at[0, 'AG'] = ''
    df.at[0, 'AG'] = ''
    df.at[0, 'AG'] = ''
    df.at[0, 'AH'] = ''
    df.at[0, 'AI'] = ''


    df['D_PATT_COLOR'] = ''

    df['D_PATT_COLOR'] = np.select(
        [
            (df['D_PATT'] == '2+2'),
            (df['D_PATT'] == '3+1'),
            (df['CLOSE'] < df['JWD']),
            (df['CLOSE'] > df['JGD'].shift(1)),
            (df['CLOSE'] > df['CLOSE'].shift(1)),
        ],
        [
            'Green',
            'Red',
            'Red',
            'Green',
            'Green',
        ],
        default='Red'  # Default color if none of the conditions are met
    )

    df.loc[0, 'D_PATT_COLOR'] = ''
    df['D_PATT_COLOR'].fillna(method='ffill', inplace=True)


    df.loc[0, 'E6'] = df.loc[0, 'CLOSE']
    i = 0
    while i <= len(df):
        df.loc[1:, 'E6'] = round(df['E6'].shift(1) * (1 - 0.2857) + (df['CLOSE'] * 0.2857), 2)
        i += 1



    df.loc[0, 'E18'] = df.loc[0, 'CLOSE']
    i = 0
    while i <= len(df):
        df.loc[1:, 'E18'] = round(df['E18'].shift(1) * (1 - 0.1053) + (df['CLOSE'] * 0.1053), 2)
        i += 1

    df['CP'] = round((df['CLOSE'] + df['E6']) / 2, 2)

    df['U.TGT 1'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) / 2), 2)
    df['U.TGT 2'] = round(df['CLOSE'].shift(1) + df['RANGE'].shift(1), 2)
    df['U.TGT 3'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) * 1.5), 2)

    df['L.TGT 1'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) / 2), 2)
    df['L.TGT 2'] = round(df['CLOSE'].shift(1) - df['RANGE'].shift(1), 2)
    df['L.TGT 3'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) * 1.5), 2)

    df['H.MCP'] = round((df['LOW'] + df['E6'].shift(1)) / 2, 2)
    df['L.MCP'] = round((df['HIGH'] + df['E6'].shift(1)) / 2, 2)


    df['FROM'] = df['FROM'].dt.strftime('%d-%b-%Y')
    df['TO'] = df['TO'].dt.strftime('%d-%b-%Y')

    select_cols = ['FROM', 'TO', 'HIGH', 'LOW', 'CLOSE', 'RANGE', 'JGD', 'JWD', 'D_PATT', 'D_PATT_COLOR', 'E6', 'E18',
                   'CP', 'U.TGT 1', 'U.TGT 2', 'U.TGT 3', 'L.TGT 1', 'L.TGT 2', 'L.TGT 3', 'H.MCP', 'L.MCP', 'AG', 'AH', 'AI']

    new_df3 = df[select_cols].copy()
    name4 = df1.at[1, 'symbol']
    # excel_write(name4, 'Quaterly', new_df3)
    return new_df3



def half_yearly_calc(df1):

    df = pd.DataFrame()
    df1['Date'] = pd.to_datetime(df1['Date'])

    df['Date'] = df1['Date']

    df['FROM'] = df1['Date'].apply(lambda x: x.replace(month=1, day=1) if x.month <= 6 else x.replace(year=x.year, month=7, day=1))
    df['TO'] = df1['Date'].apply(lambda x: x.replace(month=6, day=30) if x.month <= 6 else x.replace(year=x.year, month=12, day=31))
    df = df.drop_duplicates(subset=['FROM', 'TO']).reset_index(drop=True)
    df.at[0, 'FROM'] = df1.at[0, 'Date']


    def calculate_values(row):
        from_date = row['FROM']
        to_date = row['TO']
        df_condition = df1[(df1['Date'] >= from_date) & (df1['Date'] <= to_date)]
        high_val = df_condition['High'].max()
        low_val = df_condition['Low'].min()
        close_val = df_condition['Close'].iloc[-1] if not df_condition.empty else None
        return pd.Series({'HIGH': high_val, 'LOW': low_val, 'CLOSE': close_val})

    result = df.apply(calculate_values, axis=1)
    df = pd.concat([df, result], axis=1)


    df['HIGH'] = df['HIGH'].fillna(0)
    df['LOW'] = df['LOW'].fillna(0)
    df['RANGE'] = df['HIGH'] - df['LOW']
    df['JGD'] = np.ceil((df['HIGH'] - (df['RANGE'] * 0.382)) / 0.1) * 0.1
    df['JWD'] = np.floor((df['LOW'] + (df['RANGE'] * 0.382)) / 0.1) * 0.1
    df['RANGE'] = df['RANGE'].fillna(0)
    df['JGD'] = df['JGD'].fillna(0)
    df['JWD'] = df['JWD'].fillna(0)

    condition_d_patt1 = (
        (df['JWD'] > df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt2 = (
        (df['JGD'] < df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt3 = ~condition_d_patt1 & ~condition_d_patt2

    df.loc[(df.index >= 1) & condition_d_patt1, 'D_PATT'] = '2+2'
    df.loc[(df.index >= 1) & condition_d_patt2, 'D_PATT'] = '3+1'
    df.loc[(df.index >= 1) & condition_d_patt3, 'D_PATT'] = '2+1'



    df['D_PATT_COLOR'] = ''
    df['D_PATT_COLOR'] = np.select(
        [
            (df['D_PATT'] == '2+2'),
            (df['D_PATT'] == '3+1'),
            (df['CLOSE'] < df['JWD']),
            (df['CLOSE'] > df['JGD'].shift(1)),
            (df['CLOSE'] > df['CLOSE'].shift(1)),
        ],
        [
            'Green',
            'Red',
            'Red',
            'Green',
            'Green',
        ],
        default='Red'  # Default color if none of the conditions are met
    )

    df.loc[0, 'D_PATT_COLOR'] = ''
    df['D_PATT_COLOR'].fillna(method='ffill', inplace=True)


    df.loc[0, 'E6'] = df.loc[0, 'CLOSE']
    i = 0
    while i < len(df):
        df.loc[1:, 'E6'] = round(df['E6'].shift(1) * (1 - 0.2857) + (df['CLOSE'] * 0.2857), 2)
        i += 1

    df.loc[0, 'E18'] = df.loc[0, 'CLOSE']
    i = 0
    while i < len(df):
        df.loc[1:, 'E18'] = round(df['E18'].shift(1) * (1 - 0.1053) + (df['CLOSE'] * 0.1053), 2)
        i += 1

    df['CP'] = round((df['CLOSE'] + df['E6']) / 2, 2)

    df['U.TGT 1'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) / 2), 2)
    df['U.TGT 2'] = round(df['CLOSE'].shift(1) + df['RANGE'].shift(1), 2)
    df['U.TGT 3'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) * 1.5), 2)

    df['L.TGT 1'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) / 2), 2)
    df['L.TGT 2'] = round(df['CLOSE'].shift(1) - df['RANGE'].shift(1), 2)
    df['L.TGT 3'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) * 1.5), 2)

    df['H.MCP'] = round((df['LOW'] + df['E6'].shift(1)) / 2, 2)
    df['L.MCP'] = round((df['HIGH'] + df['E6'].shift(1)) / 2, 2)

    df.at[15, 'AF'] = 'pre-previous'
    df.at[16, 'AF'] = 'previous'
    df.at[17, 'AF'] = 'Current'
    df.at[19, 'AF'] = 'JGD'

    df.at[0, 'AG'] = ''
    df.at[0, 'AH'] = ''
    df['FROM'] = df['FROM'].dt.strftime('%d-%b-%Y')
    df['TO'] = df['TO'].dt.strftime('%d-%b-%Y')

    select_cols = ['FROM', 'TO', 'HIGH', 'LOW', 'CLOSE', 'RANGE', 'JGD', 'JWD', 'D_PATT', 'D_PATT_COLOR', 'E6', 'E18',
                   'CP', 'U.TGT 1', 'U.TGT 2', 'U.TGT 3', 'L.TGT 1', 'L.TGT 2', 'L.TGT 3', 'H.MCP', 'L.MCP', 'AF', 'AG', 'AH']

    new_df4 = df[select_cols].copy()
    name5 = df1.at[1, 'symbol']
    # excel_write(name5, 'Half-Yearly', new_df4)
    return new_df4



def yearly_calc(df1):

    df = pd.DataFrame()

    df['FROM'] = df1['Date'].apply(lambda x: x.replace(day=1, month=1))
    df['TO'] = df1['Date'].apply(lambda x: x.replace(day=31, month=12))
    df = df.drop_duplicates(subset=['FROM', 'TO']).reset_index(drop=True)
    df.at[0, 'FROM'] = df1.at[0, 'Date']

    def calculate_values(row):
        from_date = row['FROM']
        to_date = row['TO']
        df_condition = df1[(df1['Date'] >= from_date) & (df1['Date'] <= to_date)]
        high_val = df_condition['High'].max()
        low_val = df_condition['Low'].min()
        close_val = df_condition['Close'].iloc[-1] if not df_condition.empty else None
        return pd.Series({'HIGH': high_val, 'LOW': low_val, 'CLOSE': close_val})

    result = df.apply(calculate_values, axis=1)
    df = pd.concat([df, result], axis=1)


    df['HIGH'].fillna(0, inplace=True)
    df['LOW'].fillna(0, inplace=True)

    df['RANGE'] = df['HIGH'] - df['LOW']
    df['JGD'] = np.ceil((df['HIGH'] - (df['RANGE'] * 0.382)) / 0.1) * 0.1
    df['JWD'] = np.floor((df['LOW'] + (df['RANGE'] * 0.382)) / 0.1) * 0.1

    condition_d_patt1 = (
        (df['JWD'] > df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt2 = (
        (df['JGD'] < df['JWD'].shift(1).fillna(0))
    )
    condition_d_patt3 = ~condition_d_patt1 & ~condition_d_patt2

    df['D_PATT'] = ''
    df.loc[(df.index >= 1) & condition_d_patt1, 'D_PATT'] = '2+2'
    df.loc[(df.index >= 1) & condition_d_patt2, 'D_PATT'] = '3+1'
    df.loc[(df.index >= 1) & condition_d_patt3, 'D_PATT'] = '2+1'



    df['D_PATT_COLOR'] = ''

    df['D_PATT_COLOR'] = np.select(
        [
            (df['D_PATT'] == '2+2'),
            (df['D_PATT'] == '3+1'),
            (df['CLOSE'] < df['JWD']),
            (df['CLOSE'] > df['JGD'].shift(1)),
            (df['CLOSE'] > df['CLOSE'].shift(1)),
        ],
        [
            'Green',
            'Red',
            'Red',
            'Green',
            'Green',
        ],
        default='Red'  # Default color if none of the conditions are met
    )

    df.loc[0, 'D_PATT_COLOR'] = ''
    df['D_PATT_COLOR'].fillna(method='ffill', inplace=True)


    df.loc[0, 'E6'] = df.loc[0, 'CLOSE']
    i = 0
    while i <= len(df):
        df.loc[i+1:, 'E6'] = round(df['E6'].shift(1) * (1 - 0.2857) + (df['CLOSE'] * 0.2857), 2)
        i += 1


    df.loc[0, 'E18'] = df.loc[0, 'CLOSE']
    i = 0
    while i <= len(df):
        df.loc[i+1:, 'E18'] = round(df['E18'].shift(1) * (1 - 0.1053) + (df['CLOSE'] * 0.1053), 2)
        i += 1

    df['CP'] = round((df['CLOSE'] + df['E6']) / 2, 2)

    df['U.TGT 1'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) / 2), 2)
    df['U.TGT 2'] = round(df['CLOSE'].shift(1) + df['RANGE'].shift(1), 2)
    df['U.TGT 3'] = round(df['CLOSE'].shift(1) + (df['RANGE'].shift(1) * 1.5), 2)

    df['L.TGT 1'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) / 2), 2)
    df['L.TGT 2'] = round(df['CLOSE'].shift(1) - df['RANGE'].shift(1), 2)
    df['L.TGT 3'] = round(df['CLOSE'].shift(1) - (df['RANGE'].shift(1) * 1.5), 2)

    df['H.MCP'] = round((df['LOW'] + df['E6'].shift(1)) / 2, 2)
    df['L.MCP'] = round((df['HIGH'] + df['E6'].shift(1)) / 2, 2)

    df.at[15, 'AG'] = 'pre-previous'
    df.at[16, 'AG'] = 'previous'
    df.at[17, 'AG'] = 'Current'
    df.at[19, 'AG'] = 'JGD'

    df.at[19, 'AH'] = 'JWD'
    df.at[0, 'AI'] = ''
    df['FROM'] = df['FROM'].dt.strftime('%d-%b-%Y')
    df['TO'] = df['TO'].dt.strftime('%d-%b-%Y')

    select_cols = ['FROM', 'TO', 'HIGH', 'LOW', 'CLOSE', 'RANGE', 'JGD', 'JWD', 'D_PATT', 'D_PATT_COLOR', 'E6', 'E18',
                   'CP', 'U.TGT 1', 'U.TGT 2', 'U.TGT 3', 'L.TGT 1', 'L.TGT 2', 'L.TGT 3', 'H.MCP', 'L.MCP', 'AG', 'AH', 'AI']

    new_df5 = df[select_cols].copy()
    name6 = df1.at[1, 'symbol']
    # excel_write(name6, 'Yearly', new_df5)
    return new_df5



def daily_plan_calc(df1, df2):

    length = 100
    df = pd.DataFrame(index=range(length))

    df.at[1, 'A'] = 'Scrip'
    df.at[1, 'B'] = df1.loc[1, 'symbol']
    df.at[1, 'C'] = 'Date'

    df['D'] = ''
    df.at[1, 'D'] = df1.loc[2, 'symbol'].strftime('%d-%b-%Y')

    df['E'] = ''
    df['F'] = ''
    df['G'] = ''
    df['H'] = ''
    df.at[1, 'H'] = 'DAILY TRADING PLAN'
    df['I'] = ''
    df['J'] = ''
    df['K'] = ''
    df.at[0, 'L'] = 'Vlookup Date'

    last_index = df1['Close'].last_valid_index()

    if last_index is not None:
        last_value = df1['Close'].loc[last_index]
    df.at[7, 'A'] = last_value

    df.at[6, 'A'] = df2.iloc[8, 0]
    df.at[9, 'A'] = df2.iloc[11, 0]


    df1['Range'] = pd.to_numeric(df1['Range'], errors='coerce')
    filtered_df = df1[df1['Range'] != 0].dropna(subset=['Range'])
    last_value2 = filtered_df['Range'].iloc[-1]
    df.at[8, 'A'] = last_value2

    df.at[5, 'B'] = np.ceil((0.118 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[7, 'B'] = np.ceil((df.at[8, 'A'] * 0.073 + df.at[7, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[4, 'B'] = round(df.at[5, 'B'] + df.at[7, 'B'], 2)
    df.at[6, 'B'] = round(df.at[5, 'B'] - df.at[7, 'B'], 2)
    df.at[11, 'B'] = np.floor((df.at[7, 'A'] - 0.118 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[10, 'B'] = round(df.at[11, 'B'] + df.at[7, 'B'], 2)
    df.at[12, 'B'] = round(df.at[11, 'B'] - df.at[7, 'B'], 2)

    df.at[5, 'C'] = np.ceil((0.236 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[11, 'C'] = np.floor((df.at[7, 'A'] - 0.236 * df.at[8, 'A']) / 0.1) * 0.1

    df.at[5, 'D'] = np.ceil((0.382 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[4, 'D'] = round(df.at[5, 'D'] + df.at[7, 'B'], 2)
    df.at[6, 'D'] = round(df.at[5, 'D'] - df.at[7, 'B'], 2)
    df.at[11, 'D'] = np.floor((df.at[7, 'A'] - 0.382 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[10, 'D'] = round(df.at[11, 'D'] + df.at[7, 'B'], 2)
    df.at[12, 'D'] = round(df.at[11, 'D'] - df.at[7, 'B'], 2)

    df.at[4, 'E'] = np.ceil((0.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'E'] = round(df.at[4, 'E'] + df.at[7, 'B'], 2)
    df.at[5, 'E'] = round(df.at[4, 'E'] - df.at[7, 'B'], 2)
    df.at[12, 'E'] = np.floor((df.at[7, 'A'] - 0.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[11, 'E'] = round(df.at[12, 'E'] + df.at[7, 'B'], 2)
    df.at[13, 'E'] = round(df.at[12, 'E'] - df.at[7, 'B'], 2)

    df.at[4, 'F'] = np.ceil((1.0 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'F'] = round(df.at[4, 'F'] + df.at[7, 'B'], 2)
    df.at[5, 'F'] = round(df.at[4, 'F'] - df.at[7, 'B'], 2)
    df.at[12, 'F'] = np.floor((df.at[7, 'A'] - 1.0 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[11, 'F'] = round(df.at[12, 'F'] + df.at[7, 'B'], 2)
    df.at[13, 'F'] = round(df.at[12, 'F'] - df.at[7, 'B'], 2)

    df.at[3, 'G'] = np.ceil((1.272 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[2, 'G'] = round(df.at[3, 'G'] + df.at[7, 'B'], 2)
    df.at[4, 'G'] = round(df.at[3, 'G'] - df.at[7, 'B'], 2)
    df.at[7, 'G'] = 'MSF'

    df.at[13, 'G'] = np.floor((df.at[7, 'A'] - 1.272 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'G'] = round(df.at[13, 'G'] + df.at[7, 'B'], 2)
    df.at[14, 'G'] = round(df.at[13, 'G'] - df.at[7, 'B'], 2)

    df.at[3, 'H'] = np.ceil((1.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[2, 'H'] = round(df.at[3, 'H'] + df.at[7, 'B'], 2)
    df.at[4, 'H'] = round(df.at[3, 'H'] - df.at[7, 'B'], 2)
    df.at[13, 'H'] = np.floor((df.at[7, 'A'] - 1.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'H'] = round(df.at[13, 'H'] + df.at[7, 'B'], 2)
    df.at[14, 'H'] = round(df.at[13, 'H'] - df.at[7, 'B'], 2)

    df.at[3, 'I'] = np.ceil((2.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[2, 'I'] = round(df.at[3, 'I'] + df.at[7, 'B'], 2)
    df.at[4, 'I'] = round(df.at[3, 'I'] - df.at[7, 'B'], 2)

    df.at[1, 'L'] = df1['Date'].max()                          # it takes the last available date from daily sheet
    df.at[2, 'L'] = df.at[1, 'L'] - pd.Timedelta(days=1)

    df.at[1, 'L'] = df.at[1, 'L'].strftime('%d-%b-%y')
    df.at[2, 'L'] = df.at[2, 'L'].strftime('%d-%b-%y')
    msf_co = df1.loc[df1['Date'] == df.at[1, 'L'], 'MSF_Colour'].values[0]

    df.at[7, 'I'] = msf_co
    df.at[13, 'I'] = np.floor((df.at[7, 'A'] - 2.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'I'] = round(df.at[13, 'I'] + df.at[7, 'B'], 2)
    df.at[14, 'I'] = round(df.at[13, 'I'] - df.at[7, 'B'], 2)

    df.at[3, 'J'] = np.ceil((4.236 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[2, 'J'] = round(df.at[3, 'J'] + df.at[7, 'B'], 2)
    df.at[4, 'J'] = round(df.at[3, 'J'] - df.at[7, 'B'], 2)
    df.at[13, 'J'] = np.floor((df.at[7, 'A'] - 4.236 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'J'] = round(df.at[13, 'J'] + df.at[7, 'B'], 2)
    df.at[14, 'J'] = round(df.at[13, 'J'] - df.at[7, 'B'], 2)

    df.at[6, 'K'] = 'JGD'
    df.at[2, 'L'] = pd.to_datetime(df.at[2, 'L'])
    df.at[1, 'L'] = pd.to_datetime(df.at[1, 'L'])
    try:
        jgd1 = df1.loc[df1['Date'] <= df.at[2, 'L'].strftime('%d-%b-%Y')]['JGD'].values
        df.at[7, 'K'] = jgd1[-1]

    except IndexError:
        pass
    try:
        jgd2 = df1.loc[df1['Date'] <= df.at[1, 'L'].strftime('%d-%b-%Y')]['JGD'].values
        df.at[8, 'K'] = jgd2[-1]
    except IndexError:
        pass
    try:
        jwd1 = df1.loc[df1['Date'] <= df.at[2, 'L'].strftime('%d-%b-%Y')]['JWD'].values
        df.at[7, 'L'] = jwd1[-1]
    except IndexError:
        pass
    try:
        jwd2 = df1.loc[df1['Date'] <= df.at[1, 'L'].strftime('%d-%b-%Y')]['JWD'].values
        df.at[8, 'L'] = jwd2[-1]
    except IndexError:
        pass

    df.at[9, 'K'] = 'D P'

    df.at[6, 'L'] = 'JWD'

    try:
        dpatt1 = df1.loc[df1['Date'] == df.at[1, 'L'].strftime('%d-%b-%Y')]['D_patt'].values
        df.at[9, 'L'] = dpatt1[-1]
    except IndexError:
        pass

    df.at[1, 'M'] = 'yest'
    df.at[7, 'M'] = 'yest'
    df.at[8, 'M'] = 'today'

    if df.at[9, 'L'] == '3+1':
        df.at[8, 'G'] = df.at[8, 'K']
    elif df.at[9, 'L'] == '2+2':
        df.at[8, 'G'] = df.at[7, 'K']
    else:
        df.at[8, 'G'] = df.at[8, 'K']

    try:
        df.at[9, 'G'] = min([df.at[7, 'K'], df.at[8, 'K'], df.at[7, 'L'], df.at[8, 'L']])
    except numpy.core._exceptions._UFuncNoLoopError:
        pass
    try:
        df.at[10, 'G'] = np.floor((df.at[9, 'G'] - df.at[8, 'A'] * 0.146) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[7, 'H'] = df1.loc[df1['Date'] == df.at[1, 'L'].strftime('%d-%b-%Y'), 'MSF'].values[0]
    if df.at[9, 'L'] == '3+1':
        df.at[8, 'H'] = df.at[7, 'L']
    elif df.at[9, 'L'] == '2+2':
        df.at[8, 'H'] = ''
    else:
        df.at[8, 'H'] = ''

    try:
        df.at[8, 'I'] = max([df.at[7, 'K'], df.at[8, 'K'], df.at[7, 'L'], df.at[8, 'L']])
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    if df.at[9, 'L'] == '3+1':
        df.at[9, 'I'] = ''
    elif df.at[9, 'L'] == '2+2':
        df.at[9, 'I'] = df.at[8, 'L']
    else:
        df.at[9, 'I'] = ''

    try:
        df.at[8, 'J'] = np.ceil((df.at[8, 'A'] * 0.146 + df.at[8, 'I']) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[14, 'A'] = 'Trading Phase:'
    df.at[15, 'A'] = 'RH'
    df.at[16, 'A'] = 'RL'

    df.at[18, 'B'] = 'Pre Opening Check List'

    df.at[19, 'A'] = 'Indicator'
    df.at[20, 'A'] = 'MSP'
    df.at[21, 'A'] = 'Director'
    df.at[22, 'A'] = 'Pattern'
    df.at[25, 'A'] = 'Super'
    df.at[26, 'A'] = 'Bull/Bear'
    df.at[27, 'A'] = 'Weekly'
    df.at[29, 'A'] = 'Trading'
    df.at[30, 'A'] = 'Zone'

    df.at[32, 'B'] = 'App Pre Opening Tradng Notes'

    df.at[19, 'B'] = 'Theory'
    df.at[20, 'B'] = 'Sun Rise/Sun Set'
    df.at[21, 'B'] = 'Sleeping Beauty (IC)'
    df.at[22, 'B'] = 'Gap Theory'
    df.at[23, 'B'] = 'Cut...Cut...Packup (IC)'
    df.at[24, 'B'] = 'Against Situation'
    df.at[25, 'B'] = 'Dom. Super Bull'
    df.at[26, 'B'] = 'Dom. Super Bear'
    df.at[27, 'B'] = 'Support/Resistance'
    df.at[28, 'B'] = 'Gap'

    df.at[19, 'C'] = 'Action'
    df.at[20, 'C'] = 'SC'
    df.at[22, 'C'] = 'DC'
    df.at[24, 'C'] = 'DC'
    df.at[25, 'C'] = 'DC'

    df.at[30, 'E'] = 'Time'
    df.at[31, 'E'] = 'OP'

    df.at[18, 'F'] = 'Time'
    df.at[19, 'F'] = '9:15'
    df.at[20, 'F'] = '9:30'
    df.at[21, 'F'] = '10:30'
    df.at[22, 'F'] = '11:30'
    df.at[23, 'F'] = '12:30'
    df.at[24, 'F'] = '1:30'
    df.at[25, 'F'] = '2:30'
    df.at[30, 'F'] = 'Resistance'
    df.at[31, 'F'] = 1
    df.at[32, 'F'] = 2
    df.at[33, 'F'] = 3
    df.at[34, 'F'] = 4
    df.at[35, 'F'] = 5
    df.at[36, 'F'] = 6
    df.at[37, 'F'] = 7

    df.at[19, 'G'] = '9:30'
    df.at[20, 'G'] = '10:30'
    df.at[21, 'G'] = '11:30'
    df.at[22, 'G'] = '12:30'
    df.at[23, 'G'] = '1:30'
    df.at[24, 'G'] = '2:30'
    df.at[25, 'G'] = '3:30'

    df.at[18, 'H'] = 'High'
    df.at[30, 'H'] = 7
    df.at[31, 'H'] = 6
    df.at[32, 'H'] = 5
    df.at[33, 'H'] = 4
    df.at[34, 'H'] = 3
    df.at[35, 'H'] = 2
    df.at[36, 'H'] = 1
    df.at[37, 'H'] = 'Support'

    df.at[18, 'I'] = 'Low'
    df.at[18, 'J'] = 'Close'
    df.at[36, 'J'] = 'OP'
    df.at[37, 'J'] = 'Time'

    df.at[2, 'L'] = df.at[2, 'L'].strftime('%d-%b-%y')
    df.at[1, 'L'] = df.at[1, 'L'].strftime('%d-%b-%y')
    new_df6 = df.copy()
    name7 = df1.at[1, 'symbol']
    # excel_write2(name7, 'Daily-Plan', new_df6)
    return new_df6



def weekly_plan_calc(df1, df2):

    length = 100
    df = pd.DataFrame(index=range(length))

    df.loc[1, 'A'] = 'Scrip'
    df.loc[1, 'B'] = df1.loc[1, 'symbol']
    df.loc[1, 'C'] = 'Week'

    df1.at[2, 'symbol'] = pd.to_datetime(df1.at[2, 'symbol'])
    start = df1.at[2, 'symbol'] - pd.Timedelta(days=df1.at[2, 'symbol'].weekday())
    end = start + pd.Timedelta(days=6)

    df.at[1, 'D'] = start
    df.at[1, 'E'] = 'to'
    df.at[1, 'F'] = end
    df['G'] = ''
    df['H'] = ''
    df.at[1, 'I'] = 'WEEKLY TRADING PLAN'
    df['J'] = ''
    df['K'] = ''
    df['L'] = ''

    df.at[0, 'M'] = 'Vlookup Date'
    df.at[1, 'M'] = df.at[1, 'D'] - pd.Timedelta(days=7)

    df.at[1, 'N'] = 'Monday date'

    high1 = df2.loc[(df2['to_date'] == df.at[1, 'D'].strftime('%d-%b-%Y')) & (
            df2['from_date'] == df.at[1, 'F'].strftime('%d-%b-%Y')), 'High'].values
    df.at[6, 'A'] = high1[0] if len(high1) > 0 else 0

    low1 = df2.loc[(df2['to_date'] == df.at[1, 'D'].strftime('%d-%b-%Y')) & (
            df2['from_date'] == df.at[1, 'F'].strftime('%d-%b-%Y')), 'Low'].values
    df.at[7, 'A'] = low1[0] if len(low1) > 0 else 0

    df.at[9, 'A'] = df2.loc[(df2['to_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')) | (
            df2['from_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')), 'Close'].values[0]
    # df.at[9, 'A'] = close1[-1] if len(close1) > 0 else 0

    range1 = df2.loc[(df2['to_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')) | (
            df2['from_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')), 'RANGE'].values
    df.at[10, 'A'] = range1[0] if len(range1) > 0 else 0


    df.at[8, 'A'] = np.ceil((df.at[10, 'A'] * 0.073 + df.at[9, 'A']) / 0.1) * 0.1
    df.at[11, 'A'] = np.floor((df.at[9, 'A'] - df.at[10, 'A'] * 0.073) / 0.1) * 0.1

    df.at[6, 'B'] = np.ceil((0.146 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[9, 'B'] = np.ceil((0.00073 * df.at[9, 'A']) / 0.05) * 0.05
    df.at[13, 'B'] = np.floor((df.at[9, 'A'] - 0.146 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[12, 'B'] = np.ceil(((df.at[6, 'B'] - df.at[13, 'B']) * 0.382 + df.at[13, 'B']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[7, 'B'] = np.floor(((df.at[6, 'B'] - df.at[13, 'B']) * -0.382 + df.at[6, 'B']) / 0.05) * 0.05 - df.at[9, 'B']

    df.at[6, 'C'] = np.ceil((0.236 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[13, 'C'] = np.floor((df.at[9, 'A'] - 0.236 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[5, 'B'] = np.ceil(((df.at[6, 'C'] - df.at[6, 'B']) * 0.382 + df.at[6, 'B']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[14, 'B'] = np.floor(((df.at[13, 'B'] - df.at[13, 'C']) * -0.382 + df.at[13, 'B']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[6, 'D'] = np.ceil((0.382 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[7, 'D'] = np.floor(((df.at[6, 'D'] - df.at[6, 'C']) * -0.382 + df.at[6, 'D']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[13, 'D'] = np.floor((df.at[9, 'A'] - 0.382 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[12, 'D'] = np.ceil(((df.at[13, 'C'] - df.at[13, 'D']) * 0.382 + df.at[13, 'D']) / 0.05) * 0.05 + df.at[9, 'B']

    df.at[5, 'F'] = np.ceil((0.618 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[6, 'F'] = np.floor(((df.at[5, 'F'] - df.at[6, 'D']) * -0.382 + df.at[5, 'F']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[14, 'F'] = np.floor((df.at[9, 'A'] - 0.618 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[13, 'F'] = np.ceil(((df.at[13, 'D'] - df.at[14, 'F']) * 0.382 + df.at[14, 'F']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[5, 'D'] = np.ceil(((df.at[5, 'F'] - df.at[6, 'D']) * 0.382 + df.at[6, 'D']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[14, 'D'] = np.floor(((df.at[13, 'D'] - df.at[14, 'F']) * -0.382 + df.at[13, 'D']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[5, 'G'] = np.ceil((1.0 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[6, 'G'] = np.floor(((df.at[5, 'G'] - df.at[5, 'F']) * -0.382 + df.at[5, 'G']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[14, 'G'] = np.floor((df.at[9, 'A'] - 1.0 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[13, 'G'] = np.ceil(((df.at[14, 'F'] - df.at[14, 'G']) * 0.382 + df.at[14, 'G']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[4, 'F'] = np.ceil(((df.at[5, 'G'] - df.at[5, 'F']) * 0.382 + df.at[5, 'F']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[15, 'F'] = np.floor(((df.at[14, 'F'] - df.at[14, 'G']) * -0.382 + df.at[14, 'F']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'H'] = np.ceil((1.272 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'H'] = np.floor(((df.at[4, 'H'] - df.at[5, 'G']) * -0.382 + df.at[4, 'H']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[8, 'H'] = 'MSF'
    df.at[15, 'H'] = np.floor((df.at[9, 'A'] - 1.272 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'H'] = np.ceil(((df.at[14, 'G'] - df.at[15, 'H']) * 0.382 + df.at[15, 'H']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[4, 'G'] = np.ceil(((df.at[4, 'H'] - df.at[5, 'G']) * 0.382 + df.at[5, 'G']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[15, 'G'] = np.floor(((df.at[14, 'G'] - df.at[15, 'H']) * -0.382 + df.at[14, 'G']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'I'] = np.ceil((1.618 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'I'] = np.floor(((df.at[4, 'I'] - df.at[4, 'H']) * -0.382 + df.at[4, 'I']) / 0.05) * 0.05 - df.at[9, 'B']
    msf1 = df2.loc[(df2['to_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')) | (
                df2['from_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')), 'MSF'].values[0]
    df.at[8, 'I'] = msf1
    df.at[15, 'I'] = np.floor((df.at[9, 'A'] - 1.618 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'I'] = np.ceil(((df.at[15, 'H'] - df.at[15, 'I']) * 0.382 + df.at[15, 'I']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[3, 'H'] = np.ceil(((df.at[4, 'I'] - df.at[4, 'H']) * 0.382 + df.at[4, 'H']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[16, 'H'] = np.floor(((df.at[15, 'H'] - df.at[15, 'I']) * -0.382 + df.at[15, 'H']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'J'] = np.ceil((2.618 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'J'] = np.floor(((df.at[4, 'J'] - df.at[4, 'I']) * -0.382 + df.at[4, 'J']) / 0.05) * 0.05 - df.at[9, 'B']
    msf_co1 = df2.loc[(df2['to_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')) | (
                df2['from_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')), 'MSF_COLOR'].values[0]
    df.at[8, 'J'] = msf_co1
    dpatt_co1 = df2.loc[(df2['to_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')) | (
                df2['from_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')), 'D_PATT_COLOR'].values[0]
    df.at[9, 'J'] = dpatt_co1
    df.at[15, 'J'] = np.floor((df.at[9, 'A'] - 2.618 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'J'] = np.ceil(((df.at[15, 'I'] - df.at[15, 'J']) * 0.382 + df.at[15, 'J']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[3, 'I'] = np.ceil(((df.at[4, 'J'] - df.at[4, 'I']) * 0.382 + df.at[4, 'I']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[16, 'I'] = np.floor(((df.at[15, 'I'] - df.at[15, 'J']) * -0.382 + df.at[15, 'I']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'K'] = np.ceil((4.236 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'K'] = np.floor(((df.at[4, 'K'] - df.at[4, 'J']) * -0.382 + df.at[4, 'K']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[15, 'K'] = np.floor((df.at[9, 'A'] - 4.236 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'K'] = np.ceil(((df.at[15, 'J'] - df.at[15, 'K']) * 0.382 + df.at[15, 'K']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[3, 'J'] = np.ceil(((df.at[4, 'K'] - df.at[4, 'J']) * 0.382 + df.at[4, 'J']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[16, 'J'] = np.floor(((df.at[15, 'J'] - df.at[15, 'K']) * -0.382 + df.at[15, 'J']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[8, 'L'] = 'JGD'
    day1 = df.at[1, 'M'] - pd.Timedelta(days=7)
    day1_str = day1.strftime('%d-%b-%Y')
    df2['to_date'] = pd.to_datetime(df2['to_date'], format='%d-%b-%Y')
    df2['from_date'] = pd.to_datetime(df2['from_date'], format='%d-%b-%Y')

    try:
        jgd1 = df2.loc[(df2['to_date'] <= day1_str) & (df2['from_date'] >= day1_str), 'JGD'].values[0]
        df.at[9, 'L'] = jgd1
    except IndexError:
        pass

    try:
        jgd2 = df2.loc[(df2['to_date'] <= (df.at[1, 'M'].strftime('%d-%b-%Y'))) & (
                    df2['from_date'] >= (df.at[1, 'M'].strftime('%d-%b-%Y'))), 'JGD'].values[0]
        df.at[10, 'L'] = jgd2
    except IndexError:
        pass

    try:
        jwd1 = df2.loc[(df2['to_date'] <= day1_str) & (df2['from_date'] >= day1_str), 'JWD'].values[0]
        df.at[9, 'M'] = jwd1
    except IndexError:
        pass

    try:
        jwd2 = df2.loc[(df2['to_date'] <= (df.at[1, 'M'].strftime('%d-%b-%Y'))) & (
                df2['from_date'] >= (df.at[1, 'M'].strftime('%d-%b-%Y'))), 'JWD'].values[0]
        df.at[10, 'M'] = jwd2
    except IndexError:
        pass

    df.at[11, 'L'] = 'D P'

    df.at[8, 'M'] = 'JWD'

    dpatt1 = df2.loc[(df2['to_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')) | (
                df2['from_date'] == df.at[1, 'M'].strftime('%d-%b-%Y')), 'D_PATT'].values[0]
    df.at[11, 'M'] = dpatt1

    df.at[9, 'N'] = 'yest'
    df.at[10, 'N'] = 'today'

    if df.at[11, 'M'] == '3+1':
        df.at[10, 'H'] = df.at[10, 'L']
    elif df.at[11, 'M'] == '2+2':
        df.at[10, 'H'] = df.at[9, 'L']
    else:
        df.at[10, 'H'] = df.at[10, 'L']

    try:
        df.at[11, 'H'] = min([df.at[9, 'M'], df.at[10, 'M'], df.at[9, 'L'], df.at[10, 'L']])
    except TypeError:
        pass
    try:
        df.at[12, 'H'] = np.floor((df.at[11, 'H'] - df.at[10, 'A'] * 0.146) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[9, 'I'] = df.at[11, 'M']
    if df.at[11, 'M'] == '3+1':
        df.at[10, 'I'] = df.at[9, 'M']
    elif df.at[11, 'M'] == '2+2':
        df.at[10, 'I'] = ''
    else:
        df.at[10, 'I'] = ''

    try:
        if df.at[11, 'M'] == '3+1':
            df.at[10, 'J'] = df.at[9, 'L']
        elif df.at[11, 'M'] == '2+2':
            df.at[10, 'J'] = max([df.at[9, 'M'], df.at[10, 'M'], df.at[9, 'L'], df.at[10, 'L']])
        else:
            df.at[10, 'J'] = df.at[9, 'L']
    except TypeError:
        pass

    if df.at[11, 'M'] == '3+1':
        df.at[11, 'J'] = ''
    elif df.at[11, 'M'] == '2+2':
        df.at[11, 'J'] = df.at[10, 'M']
    else:
        df.at[11, 'J'] = ''
    try:
        df.at[10, 'K'] = np.ceil((df.at[10, 'A'] * 0.146 + df.at[10, 'J']) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass


    df.at[1, 'D'] = df.at[1, 'D'].strftime('%d-%b-%y')
    df.at[1, 'F'] = df.at[1, 'F'].strftime('%d-%b-%y')
    df.at[1, 'M'] = df.at[1, 'M'].strftime('%d-%b-%y')

    df.at[17, 'A'] = 'Trading Phase:'
    df.at[18, 'A'] = 'RH'
    df.at[19, 'A'] = 'RL'

    df.at[21, 'B'] = 'Pre Opening Check List'

    df.at[22, 'A'] = 'Indicator'
    df.at[23, 'A'] = 'Director'
    df.at[24, 'A'] = 'Pattern'
    df.at[27, 'A'] = 'Monthly'
    df.at[29, 'A'] = 'Trading'
    df.at[30, 'A'] = 'Zone'

    df.at[32, 'B'] = 'App Pre Opening Tradng Notes'

    df.at[22, 'B'] = 'Theory'
    df.at[23, 'B'] = 'Sleeping Beauty (IC)'
    df.at[24, 'B'] = 'Gap Theory'
    df.at[25, 'B'] = 'Cut...Cut...Packup (IC)'
    df.at[26, 'B'] = 'Against Situation'
    df.at[27, 'B'] = 'Support/Resistance'
    df.at[28, 'B'] = 'Gap'

    df.at[22, 'C'] = 'Action'
    df.at[24, 'C'] = 'DC'
    df.at[26, 'C'] = 'DC'

    df.at[32, 'D'] = 'Day'
    df.at[33, 'D'] = 'OP'

    df.at[21, 'E'] = 'Day'
    df.at[22, 'E'] = 'Monday'
    df.at[23, 'E'] = 'Tuesday'
    df.at[24, 'E'] = 'Wednesday'
    df.at[25, 'E'] = 'Thursday'
    df.at[26, 'E'] = 'Friday'
    df.at[32, 'E'] = 'Resistance'
    df.at[33, 'E'] = 1
    df.at[34, 'E'] = 2
    df.at[35, 'E'] = 3
    df.at[36, 'E'] = 4
    df.at[37, 'E'] = 5
    df.at[38, 'E'] = 6
    df.at[39, 'E'] = 7

    df.at[21, 'F'] = 'High'
    df.at[32, 'F'] = 7
    df.at[33, 'F'] = 6
    df.at[34, 'F'] = 5
    df.at[35, 'F'] = 4
    df.at[36, 'F'] = 3
    df.at[37, 'F'] = 2
    df.at[38, 'F'] = 1
    df.at[39, 'F'] = 'Support'

    df.at[21, 'G'] = 'Low'

    df.at[21, 'H'] = 'Close'
    df.at[38, 'H'] = 'OP'
    df.at[39, 'H'] = 'Day'

    new_df7 = df.copy()
    name8 = df1.at[1, 'symbol']
    # excel_write2(name8, 'Weekly-Plan', new_df7)
    return new_df7



def monthly_plan_calc(df1, df3):

    length = 100
    df = pd.DataFrame(index=range(length))

    df.loc[1, 'A'] = 'Scrip'
    df.loc[1, 'B'] = df1.loc[1, 'symbol']
    df.loc[1, 'C'] = 'Month'

    df1.at[2, 'symbol'] = pd.to_datetime(df1.at[2, 'symbol'])
    # year1 = df1.at[2, 'symbol'].year
    month_start = df1.at[2, 'symbol'].replace(day=1)
    month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
    df.at[1, 'D'] = month_start
    df.at[1, 'E'] = 'TO'
    df.at[1, 'F'] = month_end
    df['G'] = ''
    df.at[1, 'H'] = 'MONTHLY TRADING PLAN'
    df['I'] = ''
    df['J'] = ''
    df['K'] = ''
    df['L'] = ''
    df['M'] = ''
    df.at[0, 'N'] = 'Vlookup Date'
    df.at[1, 'N'] = df.at[1, 'D'] - pd.Timedelta(days=1)
    df.at[2, 'O'] = (df.at[1, 'N'] - pd.DateOffset(days=40))
    df.at[2, 'N'] = df.at[2, 'O'].strftime('%B-%y')
    df.at[1, 'D'] = df.at[1, 'D'].strftime('%d-%b-%Y')

    # df.at[7, 'A'] = df3.loc[(df3['to'] == df.at[1, 'D'].strftime('%d-%b-%Y')) | (df3['from'] == df.at[1, 'D'].strftime('%d-%b-%Y')), 'HIGH'].values[0]

    date_to_find = df.at[1, 'D'].strftime('%d-%b-%Y')
    high_values = df3[(df3['to'] == date_to_find) | (df3['from'] == date_to_find)]['HIGH'].values
    if len(high_values) > 0:
        df.at[7, 'A'] = high_values[0]
    else:
        df.at[7, 'A'] = 0

    low_values = df3[(df3['to'] == date_to_find) | (df3['from'] == date_to_find)]['LOW'].values
    if len(low_values) > 0:
        df.at[8, 'A'] = low_values[0]
    else:
        df.at[8, 'A'] = 0

    # df.at[8, 'A'] = df3.loc[(df3['to'] == df.at[1, 'D'].strftime('%d-%b-%Y')) | (df3['from'] == df.at[1, 'D'].strftime('%d-%b-%Y')), 'LOW'].values[0]

    date_to_find2 = df.at[1, 'N'].strftime('%d-%b-%Y')
    close_values = df3[(df3['to'] == date_to_find2) | (df3['from'] == date_to_find2)]['CLOSE'].values
    if len(close_values) > 0:
        df.at[9, 'A'] = close_values[0]
    else:
        df.at[9, 'A'] = 0


    range_values = df3[(df3['to'] == date_to_find2) | (df3['from'] == date_to_find2)]['RANGE'].values
    if len(range_values) > 0:
        df.at[10, 'A'] = range_values[0]
    else:
        df.at[10, 'A'] = 0

    df.at[6, 'B'] = np.ceil((0.146 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[9, 'B'] = np.ceil((0.00073 * df.at[9, 'A']) / 0.05) * 0.05
    df.at[13, 'B'] = np.floor((df.at[9, 'A'] - 0.146 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[12, 'B'] = np.ceil(((df.at[6, 'B'] - df.at[13, 'B']) * 0.382 + df.at[13, 'B']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[7, 'B'] = np.floor(((df.at[6, 'B'] - df.at[13, 'B']) * -0.382 + df.at[6, 'B']) / 0.05) * 0.05 - df.at[9, 'B']

    df.at[6, 'C'] = np.ceil((0.236 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[13, 'C'] = np.floor((df.at[9, 'A'] - 0.236 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[5, 'B'] = np.ceil(((df.at[6, 'C'] - df.at[6, 'B']) * 0.382 + df.at[6, 'B']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[14, 'B'] = np.floor(((df.at[13, 'B'] - df.at[13, 'C']) * -0.382 + df.at[13, 'B']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[6, 'D'] = np.ceil((0.382 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[7, 'D'] = np.floor(((df.at[6, 'D'] - df.at[6, 'C']) * -0.382 + df.at[6, 'D']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[13, 'D'] = np.floor((df.at[9, 'A'] - 0.382 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[12, 'D'] = np.ceil(((df.at[13, 'C'] - df.at[13, 'D']) * 0.382 + df.at[13, 'D']) / 0.05) * 0.05 + df.at[9, 'B']

    df.at[5, 'E'] = np.ceil((0.618 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[6, 'E'] = np.floor(((df.at[5, 'E'] - df.at[6, 'D']) * -0.382 + df.at[5, 'E']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[14, 'E'] = np.floor((df.at[9, 'A'] - 0.618 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[13, 'E'] = np.ceil(((df.at[13, 'D'] - df.at[14, 'E']) * 0.382 + df.at[14, 'E']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[5, 'D'] = np.ceil(((df.at[5, 'E'] - df.at[6, 'D']) * 0.382 + df.at[6, 'D']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[14, 'D'] = np.floor(((df.at[13, 'D'] - df.at[14, 'E']) * -0.382 + df.at[13, 'D']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[5, 'F'] = np.ceil((1.0 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[6, 'F'] = np.floor(((df.at[5, 'F'] - df.at[5, 'E']) * -0.382 + df.at[5, 'F']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[14, 'F'] = np.floor((df.at[9, 'A'] - 1.0 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[13, 'F'] = np.ceil(((df.at[14, 'E'] - df.at[14, 'F']) * 0.382 + df.at[14, 'F']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[4, 'E'] = np.ceil(((df.at[5, 'F'] - df.at[5, 'E']) * 0.382 + df.at[5, 'E']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[15, 'E'] = np.floor(((df.at[14, 'E'] - df.at[14, 'F']) * -0.382 + df.at[14, 'E']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'G'] = np.ceil((1.272 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'G'] = np.floor(((df.at[4, 'G'] - df.at[5, 'F']) * -0.382 + df.at[4, 'G']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[8, 'G'] = 'MSF'
    df.at[15, 'G'] = np.floor((df.at[9, 'A'] - 1.272 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'G'] = np.ceil(((df.at[14, 'F'] - df.at[15, 'G']) * 0.382 + df.at[15, 'G']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[4, 'F'] = np.ceil(((df.at[4, 'G'] - df.at[5, 'F']) * 0.382 + df.at[5, 'F']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[15, 'F'] = np.floor(((df.at[14, 'F'] - df.at[15, 'G']) * -0.382 + df.at[14, 'F']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'H'] = np.ceil((1.618 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'H'] = np.floor(((df.at[4, 'H'] - df.at[4, 'G']) * -0.382 + df.at[4, 'H']) / 0.05) * 0.05 - df.at[9, 'B']

    msf_values = df3[(df3['to'] == date_to_find2) | (df3['from'] == date_to_find2)]['MSF'].values
    if len(msf_values) > 0:
        df.at[8, 'H'] = msf_values[0]
    else:
        df.at[8, 'H'] = 0


    df.at[15, 'H'] = np.floor((df.at[9, 'A'] - 1.618 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'H'] = np.ceil(((df.at[15, 'G'] - df.at[15, 'H']) * 0.382 + df.at[15, 'H']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[3, 'G'] = np.ceil(((df.at[4, 'H'] - df.at[4, 'G']) * 0.382 + df.at[4, 'G']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[16, 'G'] = np.floor(((df.at[15, 'G'] - df.at[15, 'H']) * -0.382 + df.at[15, 'G']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'I'] = np.ceil((2.618 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'I'] = np.floor(((df.at[4, 'I'] - df.at[4, 'H']) * -0.382 + df.at[4, 'I']) / 0.05) * 0.05 - df.at[9, 'B']
    msf_color_values = df3[(df3['to'] == date_to_find2) | (df3['from'] == date_to_find2)]['MSF_COLOR'].values
    if len(msf_color_values) > 0:
        df.at[8, 'I'] = msf_color_values[0]
    else:
        df.at[8, 'I'] = 0


    d_patt_color_values = df3[(df3['to'] == date_to_find2) | (df3['from'] == date_to_find2)]['D_PATT_COLOR'].values
    if len(d_patt_color_values) > 0:
        df.at[9, 'I'] = d_patt_color_values[0]
    else:
        df.at[9, 'I'] = 0

    df.at[15, 'I'] = np.floor((df.at[9, 'A'] - 2.618 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'I'] = np.ceil(((df.at[15, 'H'] - df.at[15, 'I']) * 0.382 + df.at[15, 'I']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[3, 'H'] = np.ceil(((df.at[4, 'I'] - df.at[4, 'H']) * 0.382 + df.at[4, 'H']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[16, 'H'] = np.floor(((df.at[15, 'H'] - df.at[15, 'I']) * -0.382 + df.at[15, 'H']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[4, 'J'] = np.ceil((4.236 * df.at[10, 'A'] + df.at[9, 'A']) / 0.1) * 0.1
    df.at[5, 'J'] = np.floor(((df.at[4, 'J'] - df.at[4, 'I']) * -0.382 + df.at[4, 'J']) / 0.05) * 0.05 - df.at[9, 'B']
    df.at[15, 'J'] = np.floor((df.at[9, 'A'] - 4.236 * df.at[10, 'A']) / 0.1) * 0.1
    df.at[14, 'J'] = np.ceil(((df.at[15, 'I'] - df.at[15, 'J']) * 0.382 + df.at[15, 'J']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[3, 'I'] = np.ceil(((df.at[4, 'J'] - df.at[4, 'I']) * 0.382 + df.at[4, 'I']) / 0.05) * 0.05 + df.at[9, 'B']
    df.at[16, 'I'] = np.floor(((df.at[15, 'I'] - df.at[15, 'J']) * -0.382 + df.at[15, 'I']) / 0.05) * 0.05 - df.at[
        9, 'B']

    df.at[8, 'K'] = 'JGD'
    day1 = df.at[2, 'O']
    day1_str = day1.strftime('%d-%b-%Y')
    df3['to'] = pd.to_datetime(df3['to'], format='%d-%b-%Y')
    df3['from'] = pd.to_datetime(df3['from'], format='%d-%b-%Y')
    try:
        jgd1 = df3.loc[(df3['to'] <= day1_str) & (df3['from'] >= day1_str), 'JGD'].values[0]
        df.at[9, 'K'] = jgd1
    except IndexError:
        pass
    try:
        jgd2 = df3.loc[(df3['to'] <= (df.at[1, 'N'].strftime('%d-%b-%Y'))) & (
                df3['from'] >= (df.at[1, 'N'].strftime('%d-%b-%Y'))), 'JGD'].values[0]
        df.at[10, 'K'] = jgd2
    except IndexError:
        pass
    try:
        jwd1 = df3.loc[(df3['to'] <= day1_str) & (df3['from'] >= day1_str), 'JWD'].values[0]
        df.at[9, 'L'] = jwd1
    except IndexError:
        pass
    try:
        jwd2 = df3.loc[(df3['to'] <= (df.at[1, 'N'].strftime('%d-%b-%Y'))) & (
            df3['from'] >= (df.at[1, 'N'].strftime('%d-%b-%Y'))), 'JWD'].values[0]
        df.at[10, 'L'] = jwd2
    except IndexError:
        pass


    df.at[11, 'K'] = 'D P'
    df.at[8, 'L'] = 'JWD'



    dpatt1_values = df3[(df3['to'] == date_to_find2) | (df3['from'] == date_to_find2)]['D_PATT'].values
    if len(dpatt1_values) > 0:
        df.at[11, 'L'] = dpatt1_values[0]
    else:
        df.at[11, 'L'] = 0


    df.at[9, 'M'] = 'yest'
    df.at[10, 'M'] = 'today'

    if df.at[11, 'L'] == '3+1':
        df.at[10, 'G'] = df.at[10, 'K']
    elif df.at[11, 'L'] == '2+2':
        df.at[10, 'G'] = df.at[9, 'K']
    else:
        df.at[10, 'G'] = df.at[10, 'K']

    df.at[11, 'G'] = min([df.at[9, 'K'], df.at[10, 'K'], df.at[9, 'L'], df.at[10, 'L']])
    try:
        df.at[12, 'G'] = np.floor((df.at[11, 'G'] - df.at[10, 'A'] * 0.146) / 0.1) * 0.1
    except TypeError:
        pass

    df.at[9, 'H'] = df.at[11, 'L']
    if df.at[11, 'L'] == '3+1':
        df.at[10, 'H'] = df.at[9, 'L']
    elif df.at[11, 'L'] == '2+2':
        df.at[10, 'H'] = ''
    else:
        df.at[10, 'H'] = ''

    if df.at[11, 'L'] == '3+1':
        df.at[10, 'I'] = df.at[9, 'K']
    elif df.at[11, 'L'] == '2+2':
        df.at[10, 'I'] = max([df.at[9, 'K'], df.at[10, 'K'], df.at[9, 'L'], df.at[10, 'L']])
    else:
        df.at[10, 'I'] = df.at[9, 'K']

    if df.at[11, 'L'] == '3+1':
        df.at[11, 'I'] = ''
    elif df.at[11, 'L'] == '2+2':
        df.at[11, 'I'] = df.at[10, 'L']
    else:
        df.at[11, 'I'] = ''

    try:
        df.at[10, 'J'] = np.ceil((df.at[10, 'A'] * 0.146 + df.at[10, 'I']) / 0.1) * 0.1
    except TypeError:
        pass

    df.at[1, 'D'] = df.at[1, 'D'].strftime('%d-%b-%y')
    df.at[1, 'F'] = df.at[1, 'F'].strftime('%d-%b-%y')
    df.at[1, 'N'] = df.at[1, 'N'].strftime('%B-%y')

    df.loc[2, 'O'] = df.loc[2, 'O'].strftime('%d-%b')

    df.at[17, 'A'] = 'Trading Phase:'
    df.at[18, 'A'] = 'RH'
    df.at[19, 'A'] = 'RL'

    df.at[21, 'B'] = 'Pre Opening Check List'

    df.at[22, 'A'] = 'Indicator'
    df.at[23, 'A'] = 'Director'
    df.at[24, 'A'] = 'Pattern'
    df.at[27, 'A'] = 'Monthly'
    df.at[29, 'A'] = 'Trading'
    df.at[30, 'A'] = 'Zone'

    df.at[22, 'B'] = 'Theory'
    df.at[23, 'B'] = 'Sleeping Beauty (IC)'
    df.at[24, 'B'] = 'Gap Theory'
    df.at[25, 'B'] = 'Cut...Cut...Packup (IC)'
    df.at[26, 'B'] = 'Against Situation'
    df.at[27, 'B'] = 'Support/Resistance'
    df.at[28, 'B'] = 'Gap'

    df.at[22, 'C'] = 'Action'
    df.at[24, 'C'] = 'DC'
    df.at[26, 'C'] = 'DC'

    df.at[21, 'D'] = 'Open Trade'
    df.at[24, 'D'] = 'Day'
    df.at[25, 'D'] = 'OP'
    df.at[33, 'D'] = 'App Pre Opening Tradng Notes'

    df.at[21, 'E'] = 'High'
    df.at[24, 'E'] = 'Resistance'
    df.at[25, 'E'] = 1
    df.at[26, 'E'] = 2
    df.at[27, 'E'] = 3
    df.at[28, 'E'] = 4
    df.at[29, 'E'] = 5
    df.at[30, 'E'] = 6
    df.at[31, 'E'] = 7

    df.at[21, 'F'] = 'Low'

    df.at[21, 'G'] = 'Close'
    df.at[24, 'G'] = 7
    df.at[25, 'G'] = 6
    df.at[26, 'G'] = 5
    df.at[27, 'G'] = 4
    df.at[28, 'G'] = 3
    df.at[29, 'G'] = 2
    df.at[30, 'G'] = 1
    df.at[31, 'G'] = 'Support'

    df.at[30, 'H'] = 'OP'
    df.at[31, 'H'] = 'Day'

    new_df8 = df.copy()
    name9 = df1.at[1, 'symbol']
    # excel_write2(name9, 'Monthly-Plan', new_df8)
    return new_df8



def combined_qhy_calc(df1, df2, df4, df5, df6):

    length = 100
    df = pd.DataFrame(index=range(length))

    df.at[0, 'A'] = 'Previous'
    df.at[1, 'A'] = 'Quaterly'

    df.at[0, 'B'] = 'Quarter'
    df.at[1, 'B'] = 'Plan For :'

    df.at[0, 'C'] = ''
    df.at[0, 'D'] = ''
    df.at[0, 'E'] = df1.at[1, 'symbol']

    df1.at[2, 'symbol'] = pd.to_datetime(df1.at[2, 'symbol'])
    date1 = (df1.at[2, 'symbol'] + pd.Timedelta(days=1))
    quarter_mapping = {
        1: (pd.to_datetime(f'{date1.year}-01-01'), pd.to_datetime(f'{date1.year}-03-31')),
        2: (pd.to_datetime(f'{date1.year}-04-01'), pd.to_datetime(f'{date1.year}-06-30')),
        3: (pd.to_datetime(f'{date1.year}-07-01'), pd.to_datetime(f'{date1.year}-09-30')),
        4: (pd.to_datetime(f'{date1.year}-10-01'), pd.to_datetime(f'{date1.year}-12-31'))
    }

    quarter = (date1.month - 1) // 3 + 1
    quarter_start, quarter_end = quarter_mapping[quarter]
    df.at[1, 'C'] = quarter_start
    day_d1 = df.at[1, 'C'] - pd.Timedelta(days=1)
    df.at[1, 'D'] = quarter_end
    df.at[0, 'D'] = df.at[1, 'C'] - pd.Timedelta(days=1)

    df.at[0, 'D'] = pd.to_datetime(df.at[0, 'D'])

    try:
        df.at[5, 'A'] = df4.loc[(df4['FROM'] == df.at[1, 'C'].strftime('%d-%b-%Y')) | (df4['TO'] == df.at[1, 'C'].strftime('%d-%b-%Y')), 'HIGH'].values[0]
    except IndexError:
        df.at[5, 'A'] = 0

    try:
        df.at[6, 'A'] = df4.loc[(df4['FROM'] == df.at[1, 'C'].strftime('%d-%b-%Y')) | (df4['TO'] == df.at[1, 'C'].strftime('%d-%b-%Y')), 'LOW'].values[0]
    except IndexError:
        df.at[6, 'A'] = 0

    try:
        df.at[7, 'A'] = df4.loc[(df4['FROM'] == df.at[0, 'D'].strftime('%d-%b-%Y')) | (df4['TO'] == df.at[0, 'D'].strftime('%d-%b-%Y')), 'CLOSE'].values[0]
    except IndexError:
        df.at[6, 'A'] = 0

    try:
        df.at[8, 'A'] = df4.loc[(df4['FROM'] == df.at[0, 'D'].strftime('%d-%b-%Y')) | (df4['TO'] == df.at[0, 'D'].strftime('%d-%b-%Y')), 'RANGE'].values[0]
    except IndexError:
        df.at[8, 'A'] = 0


    df.at[12, 'A'] = ''

    target_date = df.at[1, 'C'] - pd.Timedelta(days=1)
    date_condition = (df1['Date'] <= target_date)
    if date_condition.any():
        closest_date = df1.loc[date_condition, 'Date'].max()
        df.at[14, 'A'] = closest_date
    else:
        df.at[14, 'A'] = None


    df.at[15, 'A'] = df.at[1, 'C']
    df.at[17, 'A'] = 'Previous'
    df.at[18, 'A'] = 'Half Yearly'

    df.at[17, 'B'] = 'Half year'
    df.at[18, 'B'] = 'Plan For : '

    if date1.month <= 6:
        half_year_start = date1.replace(month=1, day=1)
        half_year_end = date1.replace(month=6, day=30)
    else:
        half_year_start = date1.replace(month=7, day=1)
        half_year_end = date1.replace(month=12, day=31)

    df.at[18, 'D'] = half_year_end
    df.at[18, 'C'] = half_year_start
    df.at[17, 'D'] = (df.at[18, 'C'] - pd.Timedelta(days=1))

    try:
        df.at[22, 'A'] = df5.loc[(df5['FROM'] == df.at[18, 'C'].strftime('%d-%b-%Y')) | (
                df5['TO'] == df.at[18, 'C'].strftime('%d-%b-%Y')), 'HIGH'].values[0]
    except IndexError:
        pass

    try:
        df.at[23, 'A'] = df5.loc[(df5['FROM'] == df.at[18, 'C'].strftime('%d-%b-%Y')) | (
                df5['TO'] == df.at[18, 'C'].strftime('%d-%b-%Y')), 'LOW'].values[0]
    except IndexError:
        pass

    try:
        df.at[24, 'A'] = df5.loc[(df5['FROM'] == df.at[17, 'D'].strftime('%d-%b-%Y')) | (
                df5['TO'] == df.at[17, 'D'].strftime('%d-%b-%Y')), 'CLOSE'].values[0]
    except IndexError:
        pass

    try:
        df.at[25, 'A'] = df5.loc[(df5['FROM'] == df.at[17, 'D'].strftime('%d-%b-%Y')) | (
                df5['TO'] == df.at[17, 'D'].strftime('%d-%b-%Y')), 'RANGE'].values[0]
    except IndexError:
        pass

    df.at[29, 'A'] = ''
    df.at[30, 'A'] = ''
    date_a31 = (df.at[18, 'C'] - pd.Timedelta(days=1))

    try:
        date_a31 = df1.loc[df1['Date'] <= date_a31.strftime('%d-%b-%Y')]['Date'].values
        if len(date_a31) > 0:
            df.at[31, 'A'] = date_a31[-1]
        else:
            pass
    except ValueError:
        pass

    df.at[33, 'A'] = df.at[18, 'C']
    df.at[22, 'A'] = df.at[22, 'A'] if pd.notna(df.at[22, 'A']) else 0
    df.at[23, 'A'] = df.at[23, 'A'] if pd.notna(df.at[22, 'A']) else 0

    df.at[4, 'B'] = np.ceil((0.146 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[7, 'B'] = np.ceil((df.at[7, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[8, 'B'] = round(df.at[8, 'A'] * 7.3 / 100 + df.at[7, 'B'], 2)
    df.at[11, 'B'] = np.floor((df.at[7, 'A'] - 0.146 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[10, 'B'] = np.ceil(((df.at[4, 'B'] - df.at[11, 'B']) * 0.382 + df.at[11, 'B'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[5, 'B'] = np.floor((df.at[4, 'B'] - (df.at[4, 'B'] - df.at[11, 'B']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05

    df.at[14, 'A'] = pd.to_datetime(df.at[14, 'A'])
    try:
        date_to_find = df.at[14, 'A'].strftime('%d-%b-%Y')
        matching_dates = df1[df1['Date'] == date_to_find]

        if not matching_dates.empty:
            index_to_find = matching_dates.index[0]
            index_to_use = index_to_find + 7

            if index_to_use < len(df1):
                seven_days_ahead_date = df1.loc[index_to_use, 'Date']
                df.at[14, 'B'] = seven_days_ahead_date
            else:
                    pass
        else:
            pass
    except AttributeError:
        pass

    df.at[14, 'B'] = pd.to_datetime(df.at[14, 'B'])
    df.at[14, 'B'] = df.at[14, 'B'] + pd.Timedelta(days=11)
    df.at[21, 'B'] = np.ceil((0.146 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[24, 'B'] = np.ceil((df.at[24, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[28, 'B'] = np.floor((df.at[24, 'A'] - 0.146 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[27, 'B'] = np.ceil(
        ((df.at[21, 'B'] - df.at[28, 'B']) * 0.382 + df.at[28, 'B'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[33, 'B'] = ''

    df.at[4, 'C'] = np.ceil((0.236 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[6, 'C'] = 'TREND'
    df.at[7, 'C'] = 'LAG'
    df.at[11, 'C'] = np.floor((df.at[7, 'A'] - 0.236 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'C'] = 'BDP'
    df.at[15, 'C'] = ''
    df.at[21, 'C'] = np.ceil((0.236 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[23, 'C'] = 'TREND'
    df.at[24, 'C'] = 'LAG'
    df.at[28, 'C'] = np.floor((df.at[24, 'A'] - 0.236 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[29, 'C'] = 'BDP JGD CS 5952'
    df.at[33, 'C'] = ''
    df.at[3, 'B'] = np.ceil(((df.at[4, 'C'] - df.at[4, 'B']) * 0.382 + df.at[4, 'B'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[12, 'B'] = np.floor(
        (df.at[11, 'B'] - (df.at[11, 'B'] - df.at[11, 'C']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[20, 'B'] = np.ceil(
        ((df.at[21, 'C'] - df.at[21, 'B']) * 0.382 + df.at[21, 'B'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[29, 'B'] = np.floor(
        (df.at[28, 'B'] - (df.at[28, 'B'] - df.at[28, 'C']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[22, 'B'] = np.floor(
        (df.at[21, 'B'] - (df.at[21, 'B'] - df.at[28, 'B']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[4, 'D'] = np.ceil((0.382 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[5, 'D'] = np.floor((df.at[4, 'D'] - (df.at[4, 'D'] - df.at[4, 'C']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[6, 'D'] = 'UPTREND'
    df.at[11, 'D'] = np.floor((df.at[7, 'A'] - 0.382 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[10, 'D'] = np.ceil((df.at[11, 'D'] + (df.at[11, 'C'] - df.at[11, 'D']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05
    df.at[13, 'D'] = 'JWD'
    df.at[15, 'D'] = ''

    df.at[21, 'D'] = np.ceil((0.382 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[22, 'D'] = np.floor(
        (df.at[21, 'D'] - (df.at[21, 'D'] - df.at[21, 'C']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[23, 'D'] = 'UPTREND'
    df.at[28, 'D'] = np.floor((df.at[24, 'A'] - 0.382 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[27, 'D'] = np.ceil(
        (df.at[28, 'D'] + (df.at[28, 'C'] - df.at[28, 'D']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05
    df.at[30, 'D'] = 'WDP'
    df.at[33, 'D'] = ''

    df.at[3, 'E'] = np.ceil((0.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[4, 'E'] = np.floor((df.at[3, 'E'] - (df.at[3, 'E'] - df.at[4, 'D']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[12, 'E'] = np.floor((df.at[7, 'A'] - 0.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[11, 'E'] = np.ceil((df.at[12, 'E'] + (df.at[11, 'D'] - df.at[12, 'E']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05
    df.at[14, 'E'] = 'WDP'
    df.at[15, 'E'] = ''
    df.at[16, 'E'] = df.at[15, 'E']
    df.at[20, 'E'] = np.ceil((0.618 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[21, 'E'] = np.floor(
        (df.at[20, 'E'] - (df.at[20, 'E'] - df.at[21, 'D']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[29, 'E'] = np.floor((df.at[24, 'A'] - 0.618 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[28, 'E'] = np.ceil(
        (df.at[29, 'E'] + (df.at[28, 'D'] - df.at[29, 'E']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05
    df.at[33, 'E'] = ''
    df.at[3, 'D'] = np.ceil(((df.at[3, 'E'] - df.at[4, 'D']) * 0.382 + df.at[4, 'D'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[12, 'D'] = np.floor(
        (df.at[11, 'D'] - (df.at[11, 'D'] - df.at[12, 'E']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[20, 'D'] = np.ceil(
        ((df.at[20, 'E'] - df.at[21, 'D']) * 0.382 + df.at[21, 'D'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[29, 'D'] = np.floor(
        (df.at[28, 'D'] - (df.at[28, 'D'] - df.at[29, 'E']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[3, 'F'] = np.ceil((1.0 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[4, 'F'] = np.floor((df.at[3, 'F'] - (df.at[3, 'F'] - df.at[3, 'E']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[12, 'F'] = np.floor((df.at[7, 'A'] - 1.0 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[11, 'F'] = np.ceil((df.at[12, 'F'] + (df.at[12, 'E'] - df.at[12, 'F']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05
    df.at[15, 'F'] = 'Open Tone:'
    df.at[20, 'F'] = np.ceil((1.0 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[21, 'F'] = np.floor(
        (df.at[20, 'F'] - (df.at[20, 'F'] - df.at[20, 'E']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[29, 'F'] = np.floor((df.at[24, 'A'] - 1.0 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[28, 'F'] = np.ceil(
        (df.at[29, 'F'] + (df.at[29, 'E'] - df.at[29, 'F']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05
    df.at[33, 'F'] = 'Open Tone:'
    df.at[2, 'E'] = np.ceil(((df.at[3, 'F'] - df.at[3, 'E']) * 0.382 + df.at[3, 'E'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[13, 'E'] = np.floor(
        (df.at[12, 'E'] - (df.at[12, 'E'] - df.at[12, 'F']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[19, 'E'] = np.ceil(
        ((df.at[20, 'F'] - df.at[20, 'E']) * 0.382 + df.at[20, 'E'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[30, 'E'] = np.floor(
        (df.at[29, 'E'] - (df.at[29, 'E'] - df.at[29, 'F']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[2, 'G'] = np.ceil((1.272 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'G'] = np.floor((df.at[2, 'G'] - (df.at[2, 'G'] - df.at[3, 'F']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[4, 'G'] = 'Current Supp/Resis :'
    df.at[5, 'G'] = 'Climbing Buy/sell :'
    df.at[7, 'G'] = ''
    df.at[8, 'G'] = ''
    df.at[9, 'G'] = ''
    df.at[13, 'G'] = np.floor((df.at[7, 'A'] - 1.272 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'G'] = np.ceil((df.at[13, 'G'] + (df.at[12, 'F'] - df.at[13, 'G']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05

    df.at[19, 'G'] = np.ceil((1.272 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[20, 'G'] = np.floor(
        (df.at[19, 'G'] - (df.at[19, 'G'] - df.at[20, 'F']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[21, 'G'] = 'Current Supp/Resis :'
    df.at[22, 'G'] = 'Climbing Buy/sell :'
    df.at[24, 'G'] = ''
    df.at[25, 'G'] = ''
    df.at[26, 'G'] = ''
    df.at[30, 'G'] = np.floor((df.at[24, 'A'] - 1.272 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[29, 'G'] = np.ceil(
        (df.at[30, 'G'] + (df.at[29, 'F'] - df.at[30, 'G']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05

    df.at[2, 'F'] = np.ceil(((df.at[2, 'G'] - df.at[3, 'F']) * 0.382 + df.at[3, 'F'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[13, 'F'] = np.floor(
        (df.at[12, 'F'] - (df.at[12, 'F'] - df.at[13, 'G']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[19, 'F'] = np.ceil(
        ((df.at[19, 'G'] - df.at[20, 'F']) * 0.382 + df.at[20, 'F'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[30, 'F'] = np.floor(
        (df.at[29, 'F'] - (df.at[29, 'F'] - df.at[30, 'G']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[2, 'H'] = np.ceil((1.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'H'] = np.floor((df.at[2, 'H'] - (df.at[2, 'H'] - df.at[2, 'G']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[6, 'H'] = ''
    df.at[7, 'H'] = ''
    df.at[13, 'H'] = np.floor((df.at[7, 'A'] - 1.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'H'] = np.ceil((df.at[13, 'H'] + (df.at[13, 'G'] - df.at[13, 'H']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05
    df.at[15, 'H'] = df.at[16, 'G']
    df.at[19, 'H'] = np.ceil((1.618 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[20, 'H'] = np.floor(
        (df.at[19, 'H'] - (df.at[19, 'H'] - df.at[19, 'G']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[23, 'H'] = ''
    df.at[24, 'H'] = ''
    df.at[30, 'H'] = np.floor((df.at[24, 'A'] - 1.618 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[29, 'H'] = np.ceil(
        (df.at[30, 'H'] + (df.at[30, 'G'] - df.at[30, 'H']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05
    df.at[1, 'G'] = np.ceil(((df.at[2, 'H'] - df.at[2, 'G']) * 0.382 + df.at[2, 'G'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[14, 'G'] = np.floor(
        (df.at[13, 'G'] - (df.at[13, 'G'] - df.at[13, 'H']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[18, 'G'] = np.ceil(
        ((df.at[19, 'H'] - df.at[19, 'G']) * 0.382 + df.at[19, 'G'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[31, 'G'] = np.floor(
        (df.at[30, 'G'] - (df.at[30, 'G'] - df.at[30, 'H']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[2, 'I'] = np.ceil((2.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'I'] = np.floor((df.at[2, 'I'] - (df.at[2, 'I'] - df.at[2, 'H']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[4, 'I'] = 5884
    df.at[5, 'I'] = 6523
    df.at[6, 'I'] = ''
    df.at[7, 'I'] = ''
    df.at[8, 'I'] = ''
    df.at[13, 'I'] = np.floor((df.at[7, 'A'] - 2.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'I'] = np.ceil((df.at[13, 'I'] + (df.at[13, 'H'] - df.at[13, 'I']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05
    df.at[19, 'I'] = np.ceil((2.618 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[20, 'I'] = np.floor(
        (df.at[19, 'I'] - (df.at[19, 'I'] - df.at[19, 'H']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    df.at[21, 'I'] = 5952
    df.at[22, 'I'] = 6423
    df.at[23, 'I'] = ''
    df.at[24, 'I'] = ''
    df.at[25, 'I'] = ''
    df.at[30, 'I'] = np.floor((df.at[24, 'A'] - 2.618 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[29, 'I'] = np.ceil(
        (df.at[30, 'I'] + (df.at[30, 'H'] - df.at[30, 'I']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05
    df.at[1, 'H'] = np.ceil(((df.at[2, 'I'] - df.at[2, 'H']) * 0.382 + df.at[2, 'H'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[14, 'H'] = np.floor(
        (df.at[13, 'H'] - (df.at[13, 'H'] - df.at[13, 'I']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[18, 'H'] = np.ceil(
        ((df.at[19, 'I'] - df.at[19, 'H']) * 0.382 + df.at[19, 'H'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[31, 'H'] = np.floor(
        (df.at[30, 'H'] - (df.at[30, 'H'] - df.at[30, 'I']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[2, 'J'] = np.ceil((4.236 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'J'] = np.floor((df.at[2, 'J'] - (df.at[2, 'J'] - df.at[2, 'I']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    if df.at[7, 'I'] < df.at[7, 'G']:
        df.at[6, 'J'] = 'Bdp error'
    else:
        df.at[6, 'J'] = ''
    df.at[7, 'J'] = ''
    df.at[13, 'J'] = np.floor((df.at[7, 'A'] - 4.236 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'J'] = np.ceil((df.at[13, 'J'] + (df.at[13, 'I'] - df.at[13, 'J']) * 0.382 + df.at[7, 'B']) / 0.05) * 0.05
    df.at[19, 'J'] = np.ceil((4.236 * df.at[25, 'A'] + df.at[24, 'A']) / 0.1) * 0.1
    df.at[20, 'J'] = np.floor(
        (df.at[19, 'J'] - (df.at[19, 'J'] - df.at[19, 'I']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05
    if df.at[24, 'I'] < df.at[24, 'G']:
        df.at[23, 'J'] = 'Bdp error'
    else:
        df.at[23, 'J'] = ''
    df.at[24, 'J'] = ''
    df.at[30, 'J'] = np.floor((df.at[24, 'A'] - 4.236 * df.at[25, 'A']) / 0.1) * 0.1
    df.at[29, 'J'] = np.ceil(
        (df.at[30, 'J'] + (df.at[30, 'I'] - df.at[30, 'J']) * 0.382 + df.at[24, 'B']) / 0.05) * 0.05
    df.at[1, 'I'] = np.ceil(((df.at[2, 'J'] - df.at[2, 'I']) * 0.382 + df.at[2, 'I'] + df.at[7, 'B']) / 0.05) * 0.05
    df.at[14, 'I'] = np.floor(
        (df.at[13, 'I'] - (df.at[13, 'I'] - df.at[13, 'J']) * 0.382 - df.at[7, 'B']) / 0.05) * 0.05
    df.at[18, 'I'] = np.ceil(
        ((df.at[19, 'J'] - df.at[19, 'I']) * 0.382 + df.at[19, 'I'] + df.at[24, 'B']) / 0.05) * 0.05
    df.at[31, 'I'] = np.floor(
        (df.at[30, 'I'] - (df.at[30, 'I'] - df.at[30, 'J']) * 0.382 - df.at[24, 'B']) / 0.05) * 0.05

    df.at[31, 'A'] = pd.to_datetime(df.at[31, 'A'])

    try:
        date_b31_1 = df1.loc[df1['Date'] <= df.at[31, 'A'].strftime('%d-%b-%Y')]['Date'].values
        if len(date_b31_1) > 0:
            date_b31 = date_b31_1[-1]

        else:
            pass

        date_b31 = pd.to_datetime(date_b31)
        df.at[31, 'B'] = date_b31 + pd.Timedelta(days=20)
    except ValueError:
        pass




    df.at[35, 'A'] = 'Previous'
    df.at[36, 'A'] = 'Yearly Plan'

    df.at[35, 'B'] = 'Year'
    df.at[36, 'B'] = 'For:'
    df1.at[0, 'symbol'] = pd.to_datetime(df1.at[0, 'symbol'])
    year1 = (df1.at[0, 'symbol'] + pd.Timedelta(days=1)).year
    df.at[36, 'C'] = pd.Timestamp(year=year1, month=1, day=1)

    df.at[35, 'D'] = df.at[36, 'C'] - pd.Timedelta(days=1)
    df.at[36, 'D'] = pd.Timestamp(year=year1, month=12, day=31)

    try:
        df.at[40, 'A'] = df6.loc[(df6['FROM'] == df.at[36, 'C'].strftime('%d-%b-%Y')) | (
            df6['TO'] == df.at[36, 'C'].strftime('%d-%b-%Y')), 'HIGH'].values[0]
    except IndexError:
        df.at[40, 'A'] = 0
    try:
        df.at[41, 'A'] = df6.loc[(df6['FROM'] == df.at[36, 'C'].strftime('%d-%b-%Y')) | (
            df6['TO'] == df.at[36, 'C'].strftime('%d-%b-%Y')), 'LOW'].values[0]
    except IndexError:
        df.at[41, 'A'] = 0
    try:
        df.at[42, 'A'] = df6.loc[(df6['FROM'] == df.at[35, 'D'].strftime('%d-%b-%Y')) & (
            df6['TO'] == df.at[35, 'D'].strftime('%d-%b-%Y')), 'CLOSE'].values[0]
    except IndexError:
        df.at[42, 'A'] = 0
    try:
        df.at[43, 'A'] = df6.loc[(df6['FROM'] == df.at[35, 'D'].strftime('%d-%b-%Y')) & (
            df6['TO'] == df.at[35, 'D'].strftime('%d-%b-%Y')), 'RANGE'].values[0]
    except IndexError:
        df.at[43, 'A'] = 0

    a49 = df1.loc[df1['Date'] == (df.at[36, 'C'] - pd.Timedelta(days=1)).strftime('%d-%b-%Y'), 'Date'].values
    if len(a49) > 0:
        df.at[49, 'A'] = a49[0]
    else:
        pass

    df.at[51, 'A'] = df.at[36, 'C']

    df.at[39, 'B'] = np.ceil((0.146 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[42, 'B'] = np.ceil((df.at[42, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[46, 'B'] = np.floor((df.at[42, 'A'] - 0.146 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[40, 'B'] = np.floor(
        (df.at[39, 'B'] - (df.at[39, 'B'] - df.at[46, 'B']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[45, 'B'] = np.ceil(
        ((df.at[39, 'B'] - df.at[46, 'B']) * 0.382 + df.at[46, 'B'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[49, 'B'] = ''
    df.at[51, 'B'] = ''

    df.at[39, 'C'] = np.ceil((0.236 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[41, 'C'] = 'TREND'
    df.at[42, 'C'] = 'LEG'
    df.at[46, 'C'] = np.floor((df.at[42, 'A'] - 0.236 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[47, 'C'] = 'BDP'
    df.at[51, 'C'] = ''
    df.at[38, 'B'] = np.ceil(
        ((df.at[39, 'C'] - df.at[39, 'B']) * 0.382 + df.at[39, 'B'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[47, 'B'] = np.floor(
        (df.at[46, 'B'] - (df.at[46, 'B'] - df.at[46, 'C']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[39, 'D'] = np.ceil((0.382 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[40, 'D'] = np.floor(
        (df.at[39, 'D'] - (df.at[39, 'D'] - df.at[39, 'C']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[41, 'D'] = 'UPTREND'
    df.at[46, 'D'] = np.floor((df.at[42, 'A'] - 0.382 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[45, 'D'] = np.ceil(
        (df.at[46, 'D'] + (df.at[46, 'C'] - df.at[46, 'D']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[48, 'D'] = 'JWD'
    df.at[51, 'D'] = ''

    df.at[38, 'E'] = np.ceil((0.618 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[39, 'E'] = np.floor(
        (df.at[38, 'E'] - (df.at[38, 'E'] - df.at[39, 'D']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[47, 'E'] = np.floor((df.at[42, 'A'] - 0.618 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[46, 'E'] = np.ceil(
        (df.at[47, 'E'] + (df.at[46, 'D'] - df.at[47, 'E']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[49, 'E'] = 'WDP'
    df.at[50, 'E'] = 'CS 5379'
    df.at[51, 'E'] = ''
    df.at[38, 'D'] = np.ceil(
        ((df.at[38, 'E'] - df.at[39, 'D']) * 0.382 + df.at[39, 'D'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[47, 'D'] = np.floor(
        (df.at[46, 'D'] - (df.at[46, 'D'] - df.at[47, 'E']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[38, 'F'] = np.ceil((1.0 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[39, 'F'] = np.floor(
        (df.at[38, 'F'] - (df.at[38, 'F'] - df.at[38, 'E']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[47, 'F'] = np.floor((df.at[42, 'A'] - 1.0 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[46, 'F'] = np.ceil(
        (df.at[47, 'F'] + (df.at[47, 'E'] - df.at[47, 'F']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[51, 'F'] = 'Open Tone:'
    df.at[37, 'E'] = np.ceil(
        ((df.at[38, 'F'] - df.at[38, 'E']) * 0.382 + df.at[38, 'E'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[48, 'E'] = np.floor(
        (df.at[47, 'E'] - (df.at[47, 'E'] - df.at[47, 'F']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[37, 'G'] = np.ceil((1.272 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[38, 'G'] = np.floor(
        (df.at[37, 'G'] - (df.at[37, 'G'] - df.at[38, 'F']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[39, 'G'] = 'Current Supp/Resis :'
    df.at[40, 'G'] = 'Climbing Buy/sell :'
    df.at[42, 'G'] = ''
    df.at[43, 'G'] = ''
    df.at[44, 'G'] = ''
    df.at[48, 'G'] = np.floor((df.at[42, 'A'] - 1.272 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[47, 'G'] = np.ceil(
        (df.at[48, 'G'] + (df.at[47, 'F'] - df.at[48, 'G']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[37, 'F'] = np.ceil(
        ((df.at[37, 'G'] - df.at[38, 'F']) * 0.382 + df.at[38, 'F'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[48, 'F'] = np.floor(
        (df.at[47, 'F'] - (df.at[47, 'F'] - df.at[48, 'G']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[37, 'H'] = np.ceil((1.618 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[38, 'H'] = np.floor(
        (df.at[37, 'H'] - (df.at[37, 'H'] - df.at[37, 'G']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[41, 'H'] = ''
    df.at[42, 'H'] = ''
    df.at[48, 'H'] = np.floor((df.at[42, 'A'] - 1.618 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[47, 'H'] = np.ceil(
        (df.at[48, 'H'] + (df.at[48, 'G'] - df.at[48, 'H']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[51, 'H'] = ''
    df.at[36, 'G'] = np.ceil(
        ((df.at[37, 'H'] - df.at[37, 'G']) * 0.382 + df.at[37, 'G'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[49, 'G'] = np.floor(
        (df.at[48, 'G'] - (df.at[48, 'G'] - df.at[48, 'H']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[37, 'I'] = np.ceil((2.618 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[38, 'I'] = np.floor(
        (df.at[37, 'I'] - (df.at[37, 'I'] - df.at[37, 'H']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05
    df.at[39, 'I'] = 5379
    df.at[40, 'I'] = 6799
    df.at[48, 'I'] = np.floor((df.at[42, 'A'] - 2.618 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[47, 'I'] = np.ceil(
        (df.at[48, 'I'] + (df.at[48, 'H'] - df.at[48, 'I']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[36, 'H'] = np.ceil(
        ((df.at[37, 'I'] - df.at[37, 'H']) * 0.382 + df.at[37, 'H'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[49, 'H'] = np.floor(
        (df.at[48, 'H'] - (df.at[48, 'H'] - df.at[48, 'I']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[37, 'J'] = np.ceil((4.236 * df.at[43, 'A'] + df.at[42, 'A']) / 0.1) * 0.1
    df.at[38, 'J'] = np.floor(
        (df.at[37, 'J'] - (df.at[37, 'J'] - df.at[37, 'I']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05

    df.at[48, 'J'] = np.floor((df.at[42, 'A'] - 4.236 * df.at[43, 'A']) / 0.1) * 0.1
    df.at[47, 'J'] = np.ceil(
        (df.at[48, 'J'] + (df.at[48, 'I'] - df.at[48, 'J']) * 0.382 + df.at[42, 'B']) / 0.05) * 0.05
    df.at[36, 'I'] = np.ceil(
        ((df.at[37, 'J'] - df.at[37, 'I']) * 0.382 + df.at[37, 'I'] + df.at[42, 'B']) / 0.05) * 0.05
    df.at[49, 'I'] = np.floor(
        (df.at[48, 'I'] - (df.at[48, 'I'] - df.at[48, 'J']) * 0.382 - df.at[42, 'B']) / 0.05) * 0.05



    # quaterly
    df4.reset_index(inplace=True, drop=True)

    df4.at[14, 'AI'] = ''
    df4.at[15, 'AI'] = ''
    df4.at[16, 'AI'] = ''
    df4.at[17, 'AI'] = ''
    df4.at[18, 'AI'] = ''
    df4.at[19, 'AI'] = ''
    df4.at[20, 'AI'] = ''
    df4.at[21, 'AI'] = ''
    df4.at[22, 'AI'] = ''



    df.at[0, 'D'] = df.at[0, 'D'].strftime('%b-%y')
    df4.at[16, 'AI'] = df.at[0, 'D']
    df4.at[16, 'AI'] = pd.to_datetime(df4.at[16, 'AI'], format='%b-%y')
    df4.at[14, 'AI'] = df4.at[16, 'AI'] - pd.Timedelta(days=120)
    df4.at[14, 'AI'] = pd.to_datetime(df4.at[14, 'AI'])
    df4.at[15, 'AI'] = df4.at[16, 'AI'] - pd.DateOffset(months=3)
    df4.at[17, 'AI'] = df.at[1, 'D'].strftime('%b-%y')
    date1 = (df4.at[14, 'AI'])
    quarter_mapping = {
        1: (pd.to_datetime(f'{date1.year}-01-01'), pd.to_datetime(f'{date1.year}-03-31')),
        2: (pd.to_datetime(f'{date1.year}-04-01'), pd.to_datetime(f'{date1.year}-06-30')),
        3: (pd.to_datetime(f'{date1.year}-07-01'), pd.to_datetime(f'{date1.year}-09-30')),
        4: (pd.to_datetime(f'{date1.year}-10-01'), pd.to_datetime(f'{date1.year}-12-31'))
    }

    quarter = (date1.month - 1) // 3 + 1
    quarter_start, quarter_end = quarter_mapping[quarter]
    df4.at[15, 'AH'] = quarter_start
    df4.at[17, 'AH'] = df.at[1, 'C']

    df4.at[15, 'AG'] = 'Pre-Previous'
    df4.at[16, 'AG'] = 'previous'
    df4.at[17, 'AG'] = 'Current'
    df4.at[0, 'AH'] = ''
    df4.at[0, 'AI'] = ''

    df4['FROM'] = pd.to_datetime(df4['FROM'])
    df4['TO'] = pd.to_datetime(df4['TO'])

    df4.at[19, 'AG'] = 'JGD'
    try:
        df4.at[20, 'AG'] = df4.loc[(df4['FROM'] <= df4.at[15, 'AI']) & (df4['TO'] >= df4.at[15, 'AI']), 'JGD'].values[0]
    except IndexError:
        pass
    try:
        df4.at[21, 'AG'] = df4.loc[(df4['FROM'] <= df4.at[16, 'AI']) & (df4['TO'] >= df4.at[16, 'AI']), 'JGD'].values[0]
    except IndexError:
        pass
    df4.at[22, 'AG'] = 'DP'

    df4.at[19, 'AH'] = 'JWD'
    try:
        df4.at[20, 'AH'] = df4.loc[(df4['FROM'] <= df4.at[15, 'AI']) & (df4['TO'] >= df4.at[15, 'AI']), 'JWD'].values[0]
    except IndexError:
        pass
    try:
        df4.at[21, 'AH'] = df4.loc[(df4['FROM'] <= df4.at[16, 'AI']) & (df4['TO'] >= df4.at[16, 'AI']), 'JWD'].values[0]
    except IndexError:
        pass
    try:
        df4.at[22, 'AH'] = df4.loc[(df4['FROM'] <= df4.at[16, 'AI']) & (df4['TO'] >= df4.at[16, 'AI']), 'D_PATT'].values[0]
    except IndexError:
        pass
    df4.at[20, 'AI'] = 'yest'
    df4.at[21, 'AI'] = 'today'
    try:
        df4.at[22, 'AI'] = \
               df4.loc[(df4['FROM'] <= df4.at[16, 'AI']) & (df4['TO'] >= df4.at[16, 'AI']), 'D_PATT_COLOR'].values[0]
    except IndexError:
        pass


    df4.at[15, 'AH'] = df4.at[15, 'AH'].strftime('%b-%y')
    df4.at[14, 'AI'] = pd.to_datetime(df4.at[14, 'AI'])
    df4.at[14, 'AI'] = df4.at[14, 'AI'].strftime('%b-%y')
    df4.at[15, 'AI'] = df4.at[15, 'AI'].strftime('%b-%y')
    df4.at[17, 'AH'] = df4.at[17, 'AH'].strftime('%b-%y')
    df4.at[16, 'AI'] = df4.at[16, 'AI'].strftime('%b-%y')

    df4['FROM'] = df4['FROM'].dt.strftime('%d-%b-%y')
    df4['TO'] = df4['TO'].dt.strftime('%d-%b-%y')
    df4.sort_index(ascending=True)
    name1 = df1.at[1, 'symbol']
    # excel_write(name1, 'Quaterly', df4)





    # half-yearly
    df5.at[17, 'AG'] = df.at[18, 'C']
    df5.at[17, 'AH'] = df.at[18, 'D']
    df5.at[16, 'AH'] = df.at[17, 'D']

    df5.at[16, 'AH'] = pd.to_datetime(df5.at[16, 'AH'], format='%b-%y')
    df5.at[17, 'AG'] = pd.to_datetime(df5.at[17, 'AG'])
    df5.at[17, 'AH'] = pd.to_datetime(df5.at[17, 'AH'])

    df5.at[15, 'AG'] = df5.at[16, 'AH'] - pd.Timedelta(days=220)

    if df5.at[15, 'AG'].month <= 6:
        df5.at[15, 'AH'] = pd.Timestamp(year=df5.at[15, 'AG'].year, month=6, day=30)
    else:
        df5.at[15, 'AH'] = pd.Timestamp(year=df5.at[15, 'AG'].year, month=12, day=31)

    df5['FROM'] = pd.to_datetime(df5['FROM'])
    df5['TO'] = pd.to_datetime(df5['TO'])

    df5.at[19, 'AF'] = 'JGD'
    try:
        df5.at[20, 'AF'] = df5.loc[(df5['FROM'] <= df5.at[15, 'AH']) & (df5['TO'] >= df5.at[15, 'AH']), 'JGD'].values[0]
    except IndexError:
        pass
    try:
        df5.at[21, 'AF'] = df5.loc[(df5['FROM'] <= df5.at[16, 'AH']) & (df5['TO'] >= df5.at[16, 'AH']), 'JGD'].values[0]
    except IndexError:
        pass

    df5.at[22, 'AF'] = 'D P'

    df5.at[19, 'AG'] = 'JWD'
    try:
        df5.at[20, 'AG'] = df5.loc[(df5['FROM'] <= df5.at[15, 'AH']) & (df5['TO'] >= df5.at[15, 'AH']), 'JWD'].values[0]
    except IndexError:
        pass
    try:
        df5.at[21, 'AG'] = df5.loc[(df5['FROM'] <= df5.at[16, 'AH']) & (df5['TO'] >= df5.at[16, 'AH']), 'JWD'].values[0]
    except IndexError:
        pass
    try:
        df5.at[22, 'AG'] = df5.loc[(df5['FROM'] <= df5.at[16, 'AH']) & (df5['TO'] >= df5.at[16, 'AH']), 'D_PATT'].values[0]
    except IndexError:
        pass


    df5.at[20, 'AH'] = 'yest'
    df5.at[21, 'AH'] = 'today'

    try:
        df5.at[22, 'AH'] = \
            df5.loc[(df5['FROM'] <= df5.at[16, 'AH']) & (df5['TO'] >= df5.at[16, 'AH']), 'D_PATT_COLOR'].values[0]
    except IndexError:
        pass


    df5.at[15, 'AH'] = df5.at[15, 'AH'].strftime('%b-%y')
    df5.at[15, 'AG'] = df5.at[15, 'AG'].strftime('%b-%y')
    df5.at[16, 'AH'] = df5.at[16, 'AH'].strftime('%b-%y')
    df5.at[17, 'AG'] = df5.at[17, 'AG'].strftime('%b-%y')
    df5.at[17, 'AH'] = df5.at[17, 'AH'].strftime('%b-%y')
    df5['FROM'] = df5['FROM'].dt.strftime('%d-%b-%y')
    df5['TO'] = df5['TO'].dt.strftime('%d-%b-%y')
    # write_to_excel(df5, name1 + '_half_yearly')
    # excel_write(name1, 'Half-Yearly', df5)


    # yearly

    df6.at[17, 'AH'] = df.at[36, 'C']
    df6.at[17, 'AI'] = df.at[36, 'D']
    df6.at[16, 'AI'] = df.at[35, 'D']
    df6.at[16, 'AI'] = pd.to_datetime(df6.at[16, 'AI'], format='%b-%y')
    df6.at[15, 'AH'] = (df6.at[16, 'AI'] - pd.Timedelta(days=450)).strftime('%b-%y')

    df6.at[15, 'AH'] = pd.to_datetime(df6.at[15, 'AH'], format='%b-%y')
    year2 = df6.at[15, 'AH'].year
    df6.at[15, 'AI'] = pd.Timestamp(year=year2, month=12, day=31)

    df6['FROM'] = pd.to_datetime(df6['FROM'])
    df6['TO'] = pd.to_datetime(df6['TO'])

    filtered_vals_jgd = df6.loc[(df6['FROM'] <= df6.at[15, 'AI']) & (df6['TO'] >= df6.at[15, 'AI']), 'JGD'].values
    if len(filtered_vals_jgd) > 0:
        df6.at[20, 'AG'] = filtered_vals_jgd[0]
    else:
        pass

    try:
        jgd2 = df6.loc[(df6['FROM'] <= df6.at[16, 'AI']) & (df6['TO'] >= df6.at[16, 'AI']), 'JGD'].values[0]
        df6.at[21, 'AG'] = jgd2

    except IndexError:
        pass


    df6.at[22, 'AG'] = 'D P'

    # this logic is implemented to handle missing values
    filtered_vals_jwd = df6.loc[(df6['FROM'] <= df6.at[15, 'AI']) & (df6['TO'] >= df6.at[15, 'AI']), 'JWD'].values
    if len(filtered_vals_jwd) > 0:
        df6.at[20, 'AH'] = filtered_vals_jwd[0]
    else:
        pass

    try:
        df6.at[21, 'AH'] = df6.loc[(df6['FROM'] <= df6.at[16, 'AI']) & (df6['TO'] >= df6.at[16, 'AI']), 'JWD'].values[0]
        df6.at[22, 'AH'] = df6.loc[(df6['FROM'] <= df6.at[16, 'AI']) & (df6['TO'] >= df6.at[16, 'AI']), 'D_PATT'].values[0]
    except IndexError:
        pass
    df6.at[20, 'AI'] = 'yest'
    df6.at[21, 'AI'] = 'today'

    try:
        df6.at[22, 'AI'] = \
            df6.loc[(df6['FROM'] <= df6.at[16, 'AI']) & (df6['TO'] >= df6.at[16, 'AI']), 'D_PATT_COLOR'].values[0]
    except IndexError:
        pass

    df6.at[15, 'AI'] = df6.at[15, 'AI'].strftime('%b-%y')
    df6.at[15, 'AH'] = df6.at[15, 'AH'].strftime('%b-%y')
    df6.at[16, 'AI'] = df6.at[16, 'AI'].strftime('%b-%y')
    df6['FROM'] = df6['FROM'].dt.strftime('%d-%b-%y')
    df6['TO'] = df6['TO'].dt.strftime('%d-%b-%y')
    df6.at[17, 'AH'] = df6.at[17, 'AH'].strftime('%b-%y')
    df6.at[17, 'AI'] = df6.at[17, 'AI'].strftime('%b-%y')

    # write_to_excel(df6, name1 + '_yearly')
    # excel_write(name1, 'Yearly', df6)



    # combined continue
    if df4.at[22, 'AH'] == "3+1":
        df.at[7, 'G'] = df4.at[21, 'AG']
    elif df4.at[22, 'AH'] == "2+2":
        df.at[7, 'G'] = df4.at[20, 'AG']
    else:
        df.at[7, 'G'] = df4.at[21, 'AG']

    df.at[8, 'G'] = min([df4.at[20, 'AG'], df4.at[21, 'AG'], df4.at[20, 'AH'], df4.at[21, 'AH']])
    df.at[9, 'G'] = np.floor((df.at[8, 'G'] - df.at[8, 'A'] * 0.146) / 0.1) * 0.1

    df.at[6, 'H'] = df4.at[22, 'AH']
    if df4.at[22, 'AH'] == '3+1':
        df.at[7, 'H'] = df4.at[20, 'AH']
    elif df4.at[22, 'AH'] == '2+2':
        df.at[7, 'H'] = ''
    else:
        df.at[7, 'H'] = ''


    df.at[6, 'I'] = df4.at[22, 'AI']
    if df4.at[22, 'AH'] == '3+1':
        df.at[7, 'I'] = df4.at[20, 'AG']
    elif df4.at[22, 'AH'] == '2+2':
        df.at[7, 'I'] = max([df4.at[20, 'AG'], df4.at[21, 'AG'], df4.at[20, 'AH'], df4.at[21, 'AH']])
    else:
        df.at[7, 'I'] = df4.at[20, 'AG']


    if df4.at[22, 'AH'] == '3+1':
        df.at[8, 'I'] = ''
    elif df4.at[22, 'AH'] == '2+2':
        df.at[8, 'I'] = df4.at[21, 'AH']
    else:
        df.at[8, 'I'] = ''

    df.at[7, 'J'] = np.ceil(((df.at[8, 'A'] * 0.146) + df.at[7, 'I']) / 0.1) * 0.1

    if df5.at[22, 'AG'] == '3+1':
        df.at[24, 'G'] = df5.at[21, 'AF']
    elif df5.at[22, 'AG'] == '2+2':
        df.at[24, 'G'] = df5.at[20, 'AF']
    else:
        df.at[24, 'G'] = df5.at[21, 'AF']

    df.at[25, 'G'] = min([df5.at[20, 'AF'], df5.at[21, 'AF'], df5.at[20, 'AG'], df5.at[21, 'AG']])
    df.at[26, 'G'] = np.floor((df.at[25, 'G'] - df.at[25, 'A'] * 0.146) / 0.1) * 0.1
    df.at[6, 'G'] = ''
    df.at[23, 'H'] = df5.at[22, 'AG']
    if df5.at[22, 'AG'] == '3+1':
        df.at[24, 'H'] = df5.at[20, 'AG']
    elif df5.at[22, 'AG'] == '2+2':
        df.at[24, 'H'] = ''
    else:
        df.at[24, 'H'] = ''

    df.at[23, 'I'] = df5.at[22, 'AH']
    if df5.at[22, 'AG'] == '3+1':
        df.at[24, 'I'] = df5.at[20, 'AF']
    elif df5.at[22, 'AG'] == '2+2':
        df.at[24, 'I'] = max([df5.at[20, 'AF'], df5.at[21, 'AF'], df5.at[20, 'AG'], df5.at[21, 'AG']])
    else:
        df.at[24, 'I'] = df5.at[20, 'AF']

    if df5.at[22, 'AG'] == '3+1':
        df.at[25, 'I'] = ''
    elif df5.at[22, 'AG'] == '2+2':
        df.at[25, 'I'] = df5.at[21, 'AG']
    else:
        df.at[25, 'I'] = ''

    df.at[24, 'J'] = np.ceil(((df.at[25, 'A'] * 0.146) + df.at[24, 'I']) / 0.1) * 0.1

    if df6.at[22, 'AH'] == '3+1':
        df.at[42, 'G'] = df6.at[21, 'AG']
    elif df6.at[22, 'AH'] == '2+2':
        df.at[42, 'G'] = df6.at[20, 'AG']
    else:
        df.at[42, 'G'] = df6.at[21, 'AG']

    df.at[43, 'G'] = min([df6.at[20, 'AG'], df6.at[21, 'AG'], df6.at[20, 'AH'], df6.at[21, 'AH']])
    df.at[44, 'G'] = np.floor((df.at[43, 'G'] - df.at[43, 'A'] * 0.146) / 0.1) * 0.1

    df.at[41, 'H'] = df6.at[22, 'AH']
    if df6.at[22, 'AH'] == '3+1':
        df.at[42, 'H'] = df6.at[20, 'AH']
    elif df6.at[22, 'AH'] == '2+2':
        df.at[42, 'H'] = ''
    else:
        df.at[42, 'H'] = ''

    df.at[41, 'I'] = df6.at[22, 'AI']
    if df6.at[22, 'AH'] == '3+1':
        df.at[42, 'I'] = df6.at[20, 'AG']
    elif df6.at[22, 'AH'] == '2+2':
        df.at[42, 'I'] = max([df6.at[20, 'AG'], df6.at[21, 'AG'], df6.at[20, 'AH'], df6.at[21, 'AH']])
    else:
        df.at[42, 'I'] = df6.at[20, 'AG']

    if df.at[42, 'I'] < df.at[42, 'G']:
        df.at[41, 'J'] = 'Bdp error'
    else:
        df.at[41, 'J'] = ''

    if df6.at[22, 'AH'] == '3+1':
        df.at[43, 'I'] = 0
    elif df6.at[22, 'AH'] == '2+2':
        df.at[43, 'I'] = df6.at[21, 'AH']
    else:
        df.at[43, 'I'] = 0

    if df.at[24, 'I'] < df.at[24, 'G']:
        df.at[23, 'J'] = 'Bdp error'
    else:
        df.at[23, 'J'] = ''

    if df.at[42, 'I'] < df.at[42, 'G']:
        df.at[41, 'J'] = 'Bdp error'
    else:
        df.at[41, 'J'] = ''
    df.at[42, 'J'] = np.ceil(((df.at[43, 'A'] * 0.146) + df.at[42, 'I']) / 0.1) * 0.1

    if df.at[42, 'I'] <= df.at[39, 'B'] and df.at[42, 'I'] >= df.at[46, 'B']:
        df.at[38, 'A'] = 'BDP'
    else:
        df.at[38, 'A'] = ''
    if df.at[41, 'H'] == '2+2':
        df.at[39, 'A'] = ''
    elif df.at[42, 'G'] <= df.at[39, 'B'] and df.at[42, 'G'] >= df.at[46, 'B']:
        df.at[39, 'A'] = 'JGD'
    else:
        df.at[39, 'A'] = ''

    if df.at[43, 'I'] >= df.at[46, 'B'] and df.at[43, 'I'] <= df.at[39, 'B']:
        df.at[47, 'A'] = 'JWD'
    else:
        df.at[47, 'A'] = ''

    if df.at[43, 'G'] >= df.at[46, 'B'] and df.at[43, 'G'] <= df.at[39, 'B']:
        df.at[48, 'A'] = 'WDP'
    else:
        df.at[48, 'A'] = ''

    df.at[42, 'J'] = np.ceil(((df.at[43, 'A'] * 0.146) + df.at[42, 'I']) / 0.1) * 0.1

    days_number = 7 - (
        df.at[14, 'B'].dayofweek) - 1  # -1 because the week starts from monday, and default it starts from sunday
    try:
        df.at[15, 'B'] = df.at[14, 'B'] + pd.Timedelta(days=days_number)
    except ValueError:
        pass

    try:
        days_number2 = 7 - (df.at[31, 'B'].dayofweek) - 1
        df.at[33, 'B'] = df.at[31, 'B'] + pd.Timedelta(days=days_number2)
    except AttributeError:
        pass
    try:
        df.at[49, 'A'] = df.at[36, 'C'] - pd.Timedelta(days=2)
        df.at[49, 'B'] = df.at[49, 'A'] + pd.Timedelta(days=32)
    except AttributeError:
        pass
    try:
        days_number3 = 7 - (df.at[49, 'B'].dayofweek) - 1
        df.at[51, 'B'] = df.at[49, 'B'] + pd.Timedelta(days=days_number3)
    except AttributeError:
        pass

    df1['Date'] = pd.to_datetime(df1['Date'])
    start_date = df.at[33, 'A']
    end_date = df.at[33, 'B']
    high_val = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'High']
    low_val = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'Low']
    close_val = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'Close']
    try:
        df.at[15, 'C'] = max(high_val)
    except ValueError:
        pass
    try:
        df.at[15, 'D'] = min(low_val)
    except ValueError:
        pass
    try:
        df.at[15, 'E'] = close_val.iloc[-1]
    except IndexError:
        pass


    high_val2 = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'High']
    low_val2 = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'Low']
    close_val2 = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'Close']
    try:
        df.at[33, 'C'] = max(high_val2)
    except ValueError:
        pass
    try:
        df.at[33, 'D'] = min(low_val2)
    except ValueError:
        pass
    try:
        df.at[33, 'E'] = close_val2.iloc[-1]
    except IndexError:
        pass

    start_date2 = df.at[51, 'A'].strftime('%d-%b-%Y')
    end_date2 = df.at[51, 'B'].strftime('%d-%b-%Y')
    high_val3 = df1.loc[(df1['Date'] >= start_date2) & (df1['Date'] <= end_date2), 'High']
    low_val3 = df1.loc[(df1['Date'] >= start_date2) & (df1['Date'] <= end_date2), 'Low']
    close_val3 = df1.loc[(df1['Date'] >= start_date2) & (df1['Date'] <= end_date2), 'Close']
    try:
        df.at[51, 'C'] = max(high_val3)
    except ValueError:
        pass
    try:
        df.at[51, 'D'] = min(low_val3)
    except ValueError:
        pass
    try:
        df.at[51, 'E'] = close_val3.iloc[-1]
    except (IndexError, ValueError, AttributeError):
        pass
    try:
        if df.at[33, 'C'] >= df.at[20, 'B'] and df.at[33, 'C'] <= df.at[20, 'D'] and df.at[33, 'D'] >= df.at[29, 'D'] and \
                df.at[33, 'D'] <= df.at[29, 'B']:
            df.at[33, 'G'] = "ERROR"
        elif df.at[33, 'C'] >= df.at[20, 'D'] and df.at[33, 'C'] <= df.at[19, 'E']:
            df.at[33, 'G'] = "RMSL HIGH@RC38.2%"
        elif df.at[33, 'C'] >= df.at[20, 'D'] and df.at[33, 'C'] <= df.at[19, 'F']:
            df.at[33, 'G'] = "RMSLHIGH@RC61.8%"
        elif df.at[33, 'C'] >= df.at[19, 'E'] and df.at[33, 'C'] <= df.at[18, 'G']:
            df.at[33, 'G'] = "RMSLHIGH@RC100%"
        elif df.at[33, 'C'] >= df.at[19, 'F'] and df.at[33, 'C'] <= df.at[18, 'H']:
            df.at[33, 'G'] = "RMSL HIGH@RC127.2%"
        elif df.at[33, 'C'] >= df.at[20, 'B'] and df.at[33, 'C'] <= df.at[20, 'D']:
            df.at[33, 'G'] = "ICRC"
        elif df.at[33, 'D'] <= df.at[29, 'B'] and df.at[33, 'D'] >= df.at[29, 'D']:
            df.at[33, 'G'] = "ICFC"
        elif df.at[33, 'D'] <= df.at[29, 'D'] and df.at[33, 'D'] >= df.at[30, 'F']:
            df.at[33, 'G'] = "RMSLLOW@FC38.2%"
        elif df.at[33, 'D'] <= df.at[29, 'D'] and df.at[33, 'D'] >= df.at[30, 'F']:
            df.at[33, 'G'] = "RMSLLOW@FC61.8%"
        elif df.at[33, 'D'] <= df.at[30, 'E'] and df.at[33, 'D'] >= df.at[31, 'G']:
            df.at[33, 'G'] = "RMSLLOW@FC100%"
        elif df.at[33, 'D'] <= df.at[30, 'F'] and df.at[33, 'D'] >= df.at[31, 'H']:
            df.at[33, 'G'] = "RMSLLOW@FC127.2%"
        elif df.at[33, 'C'] <= df.at[20, 'B'] and df.at[33, 'C'] >= df.at[29, 'B'] and df.at[33, 'D'] >= df.at[29, 'B'] and \
                df.at[33, 'D'] <= df.at[20, 'B']:
            df.at[33, 'G'] = "ICRR"
        else:
            df.at[33, 'G'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[0, 'M'] = 'POR UP:'
    df.at[1, 'M'] = 'Date'
    df.at[8, 'M'] = 'T.Phase:'
    df.at[9, 'M'] = 'PCR'
    df.at[10, 'M'] = 'Indicator'
    df.at[17, 'M'] = 'POR UP:'
    df.at[18, 'M'] = 'Date'
    df.at[25, 'M'] = 'T.Phase:'
    df.at[26, 'M'] = 'PCR:'
    df.at[27, 'M'] = 'Indicators :'
    df.at[35, 'M'] = 'POR UP:'
    df.at[36, 'M'] = 'Date'
    df.at[43, 'M'] = 'T.Phase:'
    df.at[44, 'M'] = 'PCR:'
    df.at[45, 'M'] = 'Indicator:'

    df.at[0, 'N'] = df.at[6, 'A'] + df.at[8, 'A']
    df.at[1, 'N'] = 'Resistance'

    if df.at[7, 'I'] <= df.at[11, 'B'] and df.at[7, 'I'] >= df.at[11, 'D']:
        df.at[10, 'N'] = 'BSB'
    else:
        df.at[10, 'N'] = ''
    if df.at[8, 'G'] <= df.at[4, 'D'] and df.at[8, 'G'] >= df.at[4, 'B']:
        df.at[11, 'N'] = 'LG'
    else:
        df.at[11, 'N'] = ''
    if df.at[8, 'G'] >= df.at[12, 'E'] and df.at[8, 'G'] <= df.at[11, 'D']:
        df.at[12, 'N'] = 'CCP.FC'
    else:
        df.at[12, 'N'] = ''
    if df.at[8, 'G'] >= df.at[12, 'F'] and df.at[8, 'G'] <= df.at[12, 'E']:
        df.at[13, 'N'] = 'Catch.Knife'
    else:
        df.at[13, 'N'] = ''
    if df.at[7, 'I'] <= df.at[4, 'D'] and df.at[7, 'I'] >= df.at[4, 'B']:
        df.at[14, 'N'] = 'KK.BUY'
    else:
        df.at[14, 'N'] = ''

    df.at[17, 'N'] = df.at[23, 'A'] + df.at[25, 'A']
    df.at[18, 'N'] = 'Resistance'

    if df.at[24, 'I'] <= df.at[28, 'B'] and df.at[24, 'I'] >= df.at[28, 'D']:
        df.at[27, 'N'] = 'BSB'
    else:
        df.at[27, 'N'] = ''
    if df.at[25, 'G'] <= df.at[21, 'D'] and df.at[25, 'G'] >= df.at[21, 'B']:
        df.at[28, 'N'] = 'LG'
    else:
        df.at[28, 'N'] = ''
    if df.at[25, 'G'] >= df.at[29, 'E'] and df.at[25, 'G'] <= df.at[28, 'D']:
        df.at[29, 'N'] = 'CCP.FC'
    else:
        df.at[29, 'N'] = ''
    if df.at[25, 'G'] >= df.at[29, 'F'] and df.at[25, 'G'] <= df.at[29, 'E']:
        df.at[30, 'N'] = 'Catch.Knife'
    else:
        df.at[30, 'N'] = ''
    if df.at[24, 'I'] <= df.at[21, 'D'] and df.at[24, 'I'] >= df.at[21, 'B']:
        df.at[31, 'N'] = 'KK.BUY'
    else:
        df.at[31, 'N'] = ''

    df.at[35, 'N'] = df.at[41, 'A'] + df.at[43, 'A']
    df.at[36, 'N'] = 'Resistance'

    if df.at[42, 'I'] <= df.at[46, 'B'] and df.at[42, 'I'] >= df.at[46, 'D']:
        df.at[45, 'N'] = 'BSB'
    else:
        df.at[45, 'N'] = ''
    if df.at[43, 'G'] <= df.at[39, 'D'] and df.at[43, 'G'] >= df.at[39, 'B']:
        df.at[46, 'N'] = 'LG'
    else:
        df.at[46, 'N'] = ''
    if df.at[43, 'G'] >= df.at[47, 'E'] and df.at[43, 'G'] <= df.at[46, 'D']:
        df.at[47, 'N'] = 'CCP.FC'
    else:
        df.at[47, 'N'] = ''
    if df.at[43, 'G'] >= df.at[47, 'F'] and df.at[43, 'G'] <= df.at[47, 'E']:
        df.at[48, 'N'] = 'Catch.Knife'
    else:
        df.at[48, 'N'] = ''
    if df.at[42, 'I'] <= df.at[39, 'D'] and df.at[42, 'I'] >= df.at[39, 'B']:
        df.at[49, 'N'] = 'KK.Buy'
    else:
        df.at[49, 'N'] = ''

    df.at[0, 'O'] = 'POR DOWN:'
    df.at[7, 'O'] = 5764
    df.at[8, 'O'] = 'Support'
    df.at[9, 'O'] = 'PCS:'
    if df.at[7, 'I'] <= df.at[11, 'B'] and df.at[7, 'I'] >= df.at[11, 'D']:
        df.at[10, 'O'] = 'HG'
    else:
        df.at[10, 'O'] = ''
    if df.at[8, 'G'] <= df.at[4, 'D'] and df.at[8, 'G'] >= df.at[4, 'B']:
        df.at[11, 'O'] = 'RSB'
    else:
        df.at[11, 'O'] = ''
    if df.at[7, 'I'] <= df.at[3, 'E'] and df.at[7, 'I'] >= df.at[4, 'D']:
        df.at[12, 'O'] = 'CCP.RC'
    else:
        df.at[12, 'O'] = ''
    if df.at[7, 'I'] <= df.at[3, 'F'] and df.at[7, 'I'] >= df.at[3, 'E']:
        df.at[13, 'O'] = 'Cut.Kite'
    else:
        df.at[13, 'O'] = ''
    if df.at[8, 'G'] >= df.at[11, 'D'] and df.at[8, 'G'] <= df.at[11, 'B']:
        df.at[14, 'O'] = 'KK.Sell'
    else:
        df.at[14, 'O'] = ''

    df.at[16, 'C'] = df.at[15, 'C']
    df.at[16, 'D'] = df.at[15, 'D']
    df.at[16, 'E'] = df.at[15, 'E']
    try:
        if df.at[16, 'C'] >= df.at[3, 'B'] and df.at[16, 'C'] <= df.at[3, 'D'] and df.at[16, 'D'] >= df.at[12, 'D'] and \
                df.at[16, 'D'] <= df.at[12, 'B']:
            df.at[16, 'G'] = "ERROR"
        elif df.at[16, 'C'] >= df.at[3, 'D'] and df.at[16, 'C'] <= df.at[2, 'E']:
            df.at[16, 'G'] = "RMSL HIGH@RC38.2%"
        elif df.at[16, 'C'] >= df.at[3, 'D'] and df.at[16, 'C'] <= df.at[2, 'F']:
            df.at[16, 'G'] = "RMSLHIGH@RC61.8%"
        elif df.at[16, 'C'] >= df.at[2, 'E'] and df.at[16, 'C'] <= df.at[1, 'G']:
            df.at[16, 'G'] = "RMSLHIGH@RC100%"
        elif df.at[16, 'C'] >= df.at[2, 'F'] and df.at[16, 'C'] <= df.at[1, 'H']:
            df.at[16, 'G'] = "RMSL HIGH@RC127.2%"
        elif df.at[16, 'C'] >= df.at[3, 'B'] and df.at[16, 'C'] <= df.at[3, 'D']:
            df.at[16, 'G'] = "ICRC"
        elif df.at[16, 'D'] <= df.at[12, 'B'] and df.at[16, 'D'] >= df.at[12, 'D']:
            df.at[16, 'G'] = "ICFC"
        elif df.at[16, 'D'] <= df.at[12, 'D'] and df.at[16, 'D'] >= df.at[13, 'F']:
            df.at[16, 'G'] = "RMSLLOW@FC38.2%"
        elif df.at[16, 'D'] <= df.at[12, 'D'] and df.at[16, 'D'] >= df.at[13, 'F']:
            df.at[16, 'G'] = "RMSLLOW@FC61.8%"
        elif df.at[16, 'D'] <= df.at[13, 'E'] and df.at[16, 'D'] >= df.at[14, 'G']:
            df.at[16, 'G'] = "RMSLLOW@FC100%"
        elif df.at[16, 'D'] <= df.at[13, 'F'] and df.at[16, 'D'] >= df.at[14, 'H']:
            df.at[16, 'G'] = "RMSLLOW@FC127.2%"
        elif df.at[16, 'C'] <= df.at[3, 'B'] and df.at[16, 'C'] >= df.at[12, 'B'] and df.at[16, 'D'] >= df.at[12, 'B'] and \
                df.at[16, 'D'] <= df.at[3, 'B']:
            df.at[16, 'G'] = "ICRR"
        else:
            df.at[16, 'G'] = ''

        if df.at[51, 'C'] >= df.at[38, 'B'] and df.at[51, 'C'] <= df.at[38, 'D'] and df.at[51, 'D'] >= df.at[47, 'D'] and \
                df.at[51, 'D'] <= df.at[47, 'B']:
            df.at[51, 'G'] = "ERROR"
        elif df.at[51, 'C'] >= df.at[38, 'D'] and df.at[51, 'C'] <= df.at[37, 'E']:
            df.at[51, 'G'] = "RMSL HIGH@RC38.2%"
        elif df.at[51, 'C'] >= df.at[38, 'D'] and df.at[51, 'C'] <= df.at[37, 'F']:
            df.at[51, 'G'] = "RMSLHIGH@RC61.8%"
        elif df.at[51, 'C'] >= df.at[37, 'E'] and df.at[51, 'C'] <= df.at[36, 'G']:
            df.at[51, 'G'] = "RMSLHIGH@RC100%"
        elif df.at[51, 'C'] >= df.at[37, 'F'] and df.at[51, 'C'] <= df.at[36, 'H']:
            df.at[51, 'G'] = "RMSL HIGH@RC127.2%"
        elif df.at[51, 'C'] >= df.at[38, 'B'] and df.at[51, 'C'] <= df.at[38, 'D']:
            df.at[51, 'G'] = "ICRC"
        elif df.at[51, 'D'] <= df.at[47, 'B'] and df.at[51, 'D'] >= df.at[47, 'D']:
            df.at[51, 'G'] = "ICFC"
        elif df.at[51, 'D'] <= df.at[47, 'D'] and df.at[51, 'D'] >= df.at[48, 'F']:
            df.at[51, 'G'] = "RMSLLOW@FC38.2%"
        elif df.at[51, 'D'] <= df.at[47, 'D'] and df.at[51, 'D'] >= df.at[48, 'F']:
            df.at[51, 'G'] = "RMSLLOW@FC61.8%"
        elif df.at[51, 'D'] <= df.at[48, 'E'] and df.at[51, 'D'] >= df.at[49, 'G']:
            df.at[51, 'G'] = "RMSLLOW@FC100%"
        elif df.at[51, 'D'] <= df.at[48, 'F'] and df.at[51, 'D'] >= df.at[49, 'H']:
            df.at[51, 'G'] = "RMSLLOW@FC127.2%"
        elif df.at[51, 'C'] <= df.at[38, 'B'] and df.at[51, 'C'] >= df.at[47, 'B'] and df.at[51, 'D'] >= df.at[47, 'B'] and \
                df.at[51, 'D'] <= df.at[38, 'B']:
            df.at[51, 'G'] = "ICRR"
        else:
            df.at[51, 'G'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[17, 'O'] = 'POR DOWN:'
    df.at[24, 'O'] = 5860
    df.at[25, 'O'] = 'Support'
    df.at[26, 'O'] = 'PCS:'
    if df.at[24, 'I'] <= df.at[28, 'B'] and df.at[24, 'I'] >= df.at[28, 'D']:
        df.at[27, 'O'] = 'HG'
    else:
        df.at[27, 'O'] = ''
    if df.at[25, 'G'] <= df.at[21, 'D'] and df.at[25, 'G'] >= df.at[21, 'B']:
        df.at[28, 'O'] = 'RSB'
    else:
        df.at[28, 'O'] = ''
    if df.at[24, 'I'] <= df.at[20, "E"] and df.at[24, 'I'] >= df.at[21, 'D']:
        df.at[29, 'O'] = 'CCP.RC'
    else:
        df.at[29, 'O'] = ''
    if df.at[24, 'I'] <= df.at[20, 'F'] and df.at[24, 'I'] >= df.at[20, 'E']:
        df.at[30, 'O'] = 'Cut.Kite'
    else:
        df.at[30, 'O'] = ''
    if df.at[25, 'G'] >= df.at[28, 'D'] and df.at[25, 'G'] <= df.at[28, 'B']:
        df.at[31, 'O'] = 'KK.Sell'
    else:
        df.at[31, 'O'] = ''

    df.at[35, 'O'] = 'POR DOWN:'
    df.at[42, 'O'] = 5946
    df.at[43, 'O'] = 'Support'
    df.at[44, 'O'] = 'PCS'

    if df.at[42, 'I'] <= df.at[46, 'B'] and df.at[42, 'I'] >= df.at[46, 'D']:
        df.at[45, 'O'] = 'HG'
    else:
        df.at[45, 'O'] = ''
    if df.at[43, 'G'] <= df.at[39, 'D'] and df.at[43, 'G'] >= df.at[39, 'B']:
        df.at[46, 'O'] = 'RSB'
    else:
        df.at[46, 'O'] = ''
    if df.at[42, 'I'] <= df.at[38, 'E'] and df.at[42, 'I'] >= df.at[39, 'D']:
        df.at[47, 'O'] = 'CCP.RC'
    else:
        df.at[47, 'O'] = ''
    if df.at[42, 'I'] <= df.at[38, 'F'] and df.at[42, 'I'] >= df.at[38, 'E']:
        df.at[48, 'O'] = 'Cut.Kite'
    else:
        df.at[48, 'O'] = ''
    if df.at[43, 'G'] >= df.at[46, 'D'] and df.at[43, 'G'] <= df.at[46, 'B']:
        df.at[49, 'O'] = 'KK.Sell'
    else:
        df.at[49, 'O'] = ''

    df.at[0, 'P'] = df.at[5, 'A'] - df.at[8, 'A']
    df.at[7, 'P'] = 'NOW'
    df.at[8, 'P'] = 'Date'
    df.at[17, 'P'] = df.at[22, 'A'] - df.at[25, 'A']
    df.at[24, 'P'] = 'NOW'
    df.at[25, 'P'] = 'Date'
    df.at[35, 'P'] = df.at[40, 'A'] - df.at[43, 'A']
    df.at[42, 'P'] = 'NOW'
    df.at[43, 'P'] = 'Date'

    df.at[0, 'Q'] = 'Week From'
    df.at[0, 'R'] = 'To'

    df.at[1, 'Q'] = df.at[36, 'C']

    for i in range(1, len(df)):
        if df.at[i, 'Q'].weekday() <= 5:
            df.at[i, 'R'] = df.at[i, 'Q'] + pd.DateOffset(days=(6 - df.at[i, 'Q'].weekday()))
        else:
            df.at[i, 'R'] = df.at[i, 'Q']
        df.at[i + 1, 'Q'] = df.at[i, 'R'] + pd.DateOffset(days=1)

    df.at[0, 'S'] = 'High'
    df.at[0, 'T'] = 'Low'
    df.at[0, 'U'] = 'Close'

    def get_value_from_df2(df2, date_str, column_name):
        try:
            matching_rows = df2.loc[(df2['to_date'] == date_str) | (df2['from_date'] == date_str)]
            if not matching_rows.empty:
                return matching_rows.iloc[0][column_name]
        except (ValueError, IndexError):
            pass
        return None

    for y in range(len(df)):
        try:
            df.at[y, 'R'] = pd.to_datetime(df.at[y, 'R'])
            date_str = df.at[y, 'R'].strftime('%d-%b-%Y')
            df.at[y, 'S'] = get_value_from_df2(df2, date_str, 'High')
            df.at[y, 'T'] = get_value_from_df2(df2, date_str, 'Low')
            df.at[y, 'U'] = get_value_from_df2(df2, date_str, 'Close')
        except ValueError:
            pass


    df.at[0, 'V'] = ''
    df.at[0, 'W'] = ''
    df.at[8, 'X'] = 1
    df.at[9, 'X'] = 2
    df.at[10, 'X'] = 3
    df.at[11, 'X'] = 4
    df.at[12, 'X'] = 5
    df.at[13, 'X'] = 6
    df.at[14, 'X'] = 7
    df.at[15, 'X'] = 8

    df.at[8, 'Y'] = 'UPTREND CONTINUES'
    df.at[9, 'Y'] = 'PULLBACK BUY STARTS'
    df.at[10, 'Y'] = 'TREND REVERSAL BUY'
    df.at[11, 'Y'] = 'COR COMP ORD UPTREND BEGINS'
    df.at[12, 'Y'] = 'DOWNTREND CONTINUES'
    df.at[13, 'Y'] = 'CORRECTION SELL STARTS'
    df.at[14, 'Y'] = 'TREND REVERSAL SELL'
    df.at[15, 'Y'] = 'PULLBK COMP ORG DOWNTREND BEGINS'

    if df.at[24, 'I'] <= df.at[21, 'B'] and df.at[24, 'I'] >= df.at[28, 'B']:
        df.at[20, 'A'] = 'BDP'
    else:
        df.at[20, 'A'] = ''

    if df.at[23, 'H'] == '2+2':
        df.at[21, 'A'] = ''
    elif df.at[24, 'G'] <= df.at[21, 'B'] and df.at[24, 'G'] >= df.at[28, 'B']:
        df.at[21, 'A'] = 'JGD'
    else:
        df.at[21, 'A'] = ''

    try:
        for i in range(1, len(df)):
            df.at[i, 'Q'] = pd.to_datetime(df.at[i, 'Q'])
            df.at[i, 'Q'] = df.at[i, 'Q'].strftime('%d-%b-%y')
            df.at[i, 'R'] = pd.to_datetime(df.at[i, 'R'])
            df.at[i, 'R'] = df.at[i, 'R'].strftime('%d-%b-%y')
    except ValueError:  # contains missing values
        pass

    df.at[15, 'G'] = df.at[16, 'G']
    df.at[49, 'A'] = df.at[49, 'A'].strftime('%d-%b-%y')
    df.at[49, 'B'] = df.at[49, 'B'].strftime('%d-%b-%y')
    df.at[51, 'B'] = df.at[51, 'B'].strftime('%d-%b-%y')
    try:
        df.at[15, 'B'] = df.at[15, 'B'].strftime('%d-%b-%y')
    except AttributeError:
        pass
    try:
        df.at[33, 'B'] = df.at[33, 'B'].strftime('%d-%b-%y')
    except AttributeError:
        pass
    df.at[51, 'A'] = df.at[51, 'A'].strftime('%d-%b-%y')
    df.at[36, 'C'] = df.at[36, 'C'].strftime('%b-%y')
    df.at[36, 'D'] = pd.to_datetime(df.at[36, 'D'])
    df.at[36, 'D'] = df.at[36, 'D'].strftime('%b-%y')
    df.at[35, 'D'] = df.at[35, 'D'].strftime('%b-%y')
    df.at[0, 'D'] = day_d1.strftime('%b-%y')
    df.at[1, 'C'] = df.at[1, 'C'].strftime('%b-%y')
    df.at[1, 'D'] = df.at[1, 'D'].strftime('%b-%y')
    df.at[17, 'D'] = df.at[17, 'D'].strftime('%b-%y')
    df.at[18, 'D'] = df.at[18, 'D'].strftime('%d-%b-%y')
    df.at[18, 'C'] = df.at[18, 'C'].strftime('%d-%b-%y')
    df.at[33, 'A'] = df.at[33, 'A'].strftime('%d-%b-%y')
    try:
        df.at[14, 'B'] = df.at[14, 'B'].strftime('%d-%b-%y')
    except ValueError:
        pass
    try:
        df.at[14, 'A'] = df.at[14, 'A'].strftime('%d-%b-%y')
    except AttributeError:
        pass
    df.at[15, 'A'] = df.at[15, 'A'].strftime('%d-%b-%y')
    try:
        df.at[31, 'B'] = df.at[31, 'B'].strftime('%d-%b-%y')
    except AttributeError:
        pass
    try:
        df.at[31, 'A'] = df.at[31, 'A'].strftime('%d-%b-%y')
    except ValueError:
        pass

    new_df9 = df.copy()
    name10 = df1.at[1, 'symbol']
    # excel_write2(name10, 'Combined-QHY', new_df9)
    return new_df9



def combined_dwm_calc(df1, df2, df3, df4, df5, df7, df8):

    global third_date

    length = 100
    df = pd.DataFrame(index=range(length))

    df1.at[0, 'symbol'] = pd.to_datetime(df1.at[0, 'symbol'])
    df.at[0, 'A'] = df1.at[2, 'symbol'].strftime('%d-%b-%y')
    df['B'] = ''
    df.at[0, 'C'] = df4.at[1, 'D']
    df.at[0, 'D'] = df1.at[1, 'symbol']
    df.at[0, 'C'] = pd.to_datetime(df.at[0, 'C'])
    df.at[0, 'C'] = df.at[0, 'C'].strftime('%d-%b-%y')

    df.at[7, 'A'] = df4.iloc[7, 0]
    df.at[8, 'A'] = df4.iloc[8, 0]
    df.at[15, 'A'] = '06-10-2016'
    df.at[16, 'A'] = '9.15 To 9.30'
    df.at[18, 'A'] = 'Weekly'

    df.at[25, 'A'] = df5.iloc[9, 0]
    df.at[26, 'A'] = df5.iloc[10, 0]

    df.at[33, 'A'] = ''
    df.at[34, 'A'] = ''
    df.at[6, 'A'] = np.ceil((df.at[26, 'A'] * 0.073 + df.at[25, 'A']) / 0.1) * 0.1
    df.at[9, 'A'] = np.floor((df.at[25, 'A'] - df.at[26, 'A'] * 0.073) / 0.1) * 0.1

    df.at[4, 'B'] = np.ceil((0.118 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[8, 'B'] = np.ceil((df.at[8, 'A'] * 0.073 + df.at[7, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[11, 'B'] = np.floor((df.at[7, 'A'] - 0.118 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[10, 'B'] = np.ceil(((df.at[4, 'B'] - df.at[11, 'B']) * 0.382 + df.at[11, 'B']) / 0.05) * 0.05
    df.at[5, 'B'] = np.floor(((df.at[4, 'B'] - df.at[11, 'B']) * -0.382 + df.at[4, 'B']) / 0.05) * 0.05

    df.at[18, 'B'] = 'Plan For:'

    df.at[25, 'B'] = np.ceil((df.at[25, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[26, 'B'] = np.ceil((df.at[26, 'A'] * 0.073 + df.at[25, 'B']) / 0.05) * 0.05
    df.at[29, 'B'] = np.floor((df.at[25, 'A'] - 0.146 * df.at[26, 'A']) / 0.1) * 0.1


    df.at[4, 'C'] = np.ceil((0.236 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[8, 'C'] = 'LOOK FOR:'
    df.at[11, 'C'] = np.floor((df.at[7, 'A'] - 0.236 * df.at[8, 'A']) / 0.1) * 0.1

    df.at[18, 'C'] = df5.iloc[1, 3]

    df.at[22, 'C'] = np.ceil((0.236 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[24, 'C'] = 'TREND:'
    df.at[25, 'C'] = 'LEG:'
    df.at[27, 'C'] = 'LOOK FOR'
    df.at[29, 'C'] = np.floor((df.at[25, 'A'] - 0.236 * df.at[26, 'A']) / 0.1) * 0.1

    df.at[30, 'B'] = np.floor((df.at[29, 'B'] - (df.at[29, 'B'] - df.at[29, 'C']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[3, 'B'] = np.ceil(((df.at[4, 'C'] - df.at[4, 'B']) * 0.382 + df.at[4, 'B']) / 0.05) * 0.05
    df.at[12, 'B'] = np.floor(((df.at[11, 'B'] - df.at[11, 'C']) * -0.382 + df.at[11, 'B']) / 0.05) * 0.05

    df.at[4, 'D'] = np.ceil((0.382 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[5, 'D'] = np.floor(((df.at[4, 'D'] - df.at[4, 'C']) * -0.382 + df.at[4, 'D']) / 0.05) * 0.05
    df.at[11, 'D'] = np.floor((df.at[7, 'A'] - 0.382 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[10, 'D'] = np.ceil(((df.at[11, 'C'] - df.at[11, 'D']) * 0.382 + df.at[11, 'D']) / 0.05) * 0.05

    df.at[18, 'D'] = df5.at[1, 'F']
    df.at[22, 'D'] = np.ceil((0.382 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[23, 'D'] = np.floor(
        (df.at[22, 'D'] - (df.at[22, 'D'] - df.at[22, 'C']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[29, 'D'] = np.floor((df.at[25, 'A'] - 0.382 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[28, 'D'] = np.ceil(
        (df.at[29, 'D'] + (df.at[29, 'C'] - df.at[29, 'D']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05

    df.at[3, 'E'] = np.ceil((0.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[4, 'E'] = np.floor(((df.at[3, 'E'] - df.at[4, 'D']) * -0.382 + df.at[3, 'E']) / 0.05) * 0.05
    day1 = df1.at[0, 'symbol']

    df.at[6, 'E'] = df1.loc[df1['Date'] == day1.strftime('%d-%b-%Y'), 'I'].values[0]
    df.at[12, 'E'] = np.floor((df.at[7, 'A'] - 0.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[11, 'E'] = np.ceil(((df.at[11, 'D'] - df.at[12, 'E']) * 0.382 + df.at[12, 'E']) / 0.05) * 0.05

    df.at[21, 'E'] = np.ceil((0.618 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[22, 'E'] = np.floor(
        (df.at[21, 'E'] - (df.at[21, 'E'] - df.at[22, 'D']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[30, 'E'] = np.floor((df.at[25, 'A'] - 0.618 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[29, 'E'] = np.ceil(
        (df.at[30, 'E'] + (df.at[29, 'D'] - df.at[30, 'E']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05
    df.at[3, 'D'] = np.ceil(((df.at[3, 'E'] - df.at[4, 'D']) * 0.382 + df.at[4, 'D']) / 0.05) * 0.05
    df.at[12, 'D'] = np.floor(((df.at[11, 'D'] - df.at[12, 'E']) * -0.382 + df.at[11, 'D']) / 0.05) * 0.05
    df.at[21, 'D'] = np.ceil(
        ((df.at[21, 'E'] - df.at[22, 'D']) * 0.382 + df.at[22, 'D'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[30, 'D'] = np.floor(
        (df.at[29, 'D'] - (df.at[29, 'D'] - df.at[30, 'E']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[3, 'F'] = np.ceil((1.0 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[4, 'F'] = np.floor(((df.at[3, 'F'] - df.at[4, 'E']) * -0.382 + df.at[3, 'F']) / 0.05) * 0.05
    df.at[12, 'F'] = np.floor((df.at[7, 'A'] - 1.0 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[11, 'F'] = np.ceil(((df.at[12, 'E'] - df.at[12, 'F']) * 0.382 + df.at[12, 'F']) / 0.05) * 0.05

    df.at[16, 'F'] = 'Open Tone:'
    df.at[21, 'F'] = np.ceil((1.0 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[22, 'F'] = np.floor(
        (df.at[21, 'F'] - (df.at[21, 'F'] - df.at[21, 'E']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[30, 'F'] = np.floor((df.at[25, 'A'] - 1.0 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[29, 'F'] = np.ceil(
        (df.at[30, 'F'] + (df.at[30, 'E'] - df.at[30, 'F']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05

    df.at[34, 'F'] = 'Open Tone:'
    df.at[2, 'E'] = np.ceil(((df.at[3, 'F'] - df.at[3, 'E']) * 0.382 + df.at[3, 'E']) / 0.05) * 0.05
    df.at[13, 'E'] = np.floor(((df.at[12, 'E'] - df.at[12, 'F']) * -0.382 + df.at[12, 'E']) / 0.05) * 0.05
    df.at[20, 'E'] = np.ceil(
        ((df.at[21, 'F'] - df.at[21, 'E']) * 0.382 + df.at[21, 'E'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[31, 'E'] = np.floor(
        (df.at[30, 'E'] - (df.at[30, 'E'] - df.at[30, 'F']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[2, 'G'] = np.ceil((1.272 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'G'] = np.floor(((df.at[2, 'G'] - df.at[3, 'F']) * -0.382 + df.at[2, 'G']) / 0.05) * 0.05
    df.at[6, 'G'] = df4.iloc[9, 11]
    df.at[7, 'G'] = df4.iloc[8, 6]
    df.at[8, 'G'] = df4.iloc[9, 6]
    try:
        df.at[9, 'G'] = np.floor((df.at[8, 'G'] - df.at[8, 'A'] * 0.146) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[10, 'G'] = df1.loc[df1['Date'] == day1.strftime('%d-%b-%Y'), 'High'].values[0]
    df.at[13, 'G'] = np.floor((df.at[7, 'A'] - 1.272 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'G'] = np.ceil(((df.at[12, 'F'] - df.at[13, 'G']) * 0.382 + df.at[13, 'G']) / 0.05) * 0.05

    if df.at[16, 'C'] >= df.at[3, 'B'] and df.at[16, 'C'] <= df.at[3, 'D'] and df.at[16, 'D'] >= df.at[12, 'D'] and \
            df.at[16, 'D'] <= df.at[12, 'B']:
        df.at[16, 'G'] = 'ERROR'
    elif df.at[16, 'C'] >= df.at[3, 'D'] and df.at[16, 'C'] <= df.at[2, 'E']:
        df.at[16, 'G'] = 'RMSL HIGH@RC38.2%'
    elif df.at[16, 'C'] >= df.at[3, 'D'] and df.at[16, 'C'] <= df.at[2, 'F']:
        df.at[16, 'G'] = 'RMSLHIGH@RC61.8%'
    elif df.at[16, 'C'] >= df.at[2, 'E'] and df.at[16, 'C'] <= df.at[1, 'G']:
        df.at[16, 'G'] = 'RMSLHIGH@RC100%'
    elif df.at[16, 'C'] >= df.at[2, 'F'] and df.at[16, 'C'] <= df.at[1, 'H']:
        df.at[16, 'G'] = 'RMSL HIGH@RC127.2%'
    elif df.at[16, 'C'] >= df.at[3, 'B'] and df.at[16, 'C'] <= df.at[3, 'D']:
        df.at[16, 'G'] = 'ICRC'
    elif df.at[16, 'D'] <= df.at[12, 'B'] and df.at[16, 'D'] >= df.at[12, 'D']:
        df.at[16, 'G'] = 'ICFC'
    elif df.at[16, 'D'] <= df.at[12, 'D'] and df.at[16, 'D'] >= df.at[13, 'F']:
        df.at[16, 'G'] = 'RMSLLOW@FC38.2%'
    elif df.at[16, 'D'] <= df.at[12, 'D'] and df.at[16, 'D'] >= df.at[13, 'F']:
        df.at[16, 'G'] = 'RMSLLOW@FC61.8%'
    elif df.at[16, 'D'] <= df.at[13, 'E'] and df.at[16, 'D'] >= df.at[14, 'G']:
        df.at[16, 'G'] = 'RMSLLOW@FC100%'
    elif df.at[16, 'D'] <= df.at[13, 'F'] and df.at[16, 'D'] >= df.at[14, 'H']:
        df.at[16, 'G'] = 'RMSLLOW@FC127.2%'
    elif df.at[16, 'C'] <= df.at[3, 'B'] and df.at[16, 'C'] >= df.at[12, 'B'] and df.at[16, 'D'] >= df.at[12, 'B'] and \
            df.at[16, 'D'] <= df.at[3, 'B']:
        df.at[16, 'G'] = 'ICRR'
    else:
        df.at[16, 'G'] = ''

    df.at[20, 'G'] = np.ceil((1.272 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[21, 'G'] = np.floor(
        (df.at[20, 'G'] - (df.at[20, 'G'] - df.at[21, 'F']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[23, 'G'] = 'MSF'
    df.at[24, 'G'] = ''
    df.at[25, 'G'] = df5.iloc[10, 7]
    df.at[26, 'G'] = df5.iloc[11, 7]
    try:
        df.at[27, 'G'] = np.floor((df.at[26, 'G'] - df.at[26, 'A'] * 0.146) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass
    df.at[31, 'G'] = np.floor((df.at[25, 'A'] - 1.272 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[30, 'G'] = np.ceil(
        (df.at[31, 'G'] + (df.at[30, 'F'] - df.at[31, 'G']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05
    df.at[2, 'F'] = np.ceil(((df.at[2, 'G'] - df.at[3, 'F']) * 0.382 + df.at[3, 'F']) / 0.05) * 0.05
    df.at[13, 'F'] = np.floor(((df.at[12, 'F'] - df.at[13, 'G']) * -0.382 + df.at[12, 'F']) / 0.05) * 0.05
    df.at[20, 'F'] = np.ceil(
        ((df.at[20, 'G'] - df.at[21, 'F']) * 0.382 + df.at[21, 'F'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[31, 'F'] = np.floor(
        (df.at[30, 'F'] - (df.at[30, 'F'] - df.at[31, 'G']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[34, 'A'] = df.at[18, 'C']
    df.at[34, 'B'] = df.at[18, 'D']
    df.at[34, 'A'] = pd.to_datetime(df.at[34, 'A'])


    c34_high = df1.loc[df1['Date'] == df.at[34, 'A'].strftime('%d-%b-%Y'), 'High'].values
    if len(c34_high) > 0:
        df.at[34, 'C'] = c34_high[0]
    else:
        pass

    d34_low = df1.loc[df1['Date'] == df.at[34, 'A'].strftime('%d-%b-%Y'), 'Low'].values
    if len(d34_low) > 0:
        df.at[34, 'D'] = d34_low[0]
    else:
        pass

    e34_close = df1.loc[df1['Date'] == df.at[34, 'A'].strftime('%d-%b-%Y'), 'Close'].values
    if len(e34_close) > 0:
        df.at[34, 'E'] = e34_close[0]
    else:
        pass

    df.at[34, 'A'] = df.at[34, 'A'].strftime('%d-%b-%Y')

    df.at[22, 'B'] = np.ceil((0.146 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[28, 'B'] = np.ceil(((df.at[22, 'B'] - df.at[29, 'B']) * 0.382 + df.at[29, 'B'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[23, 'B'] = np.floor((df.at[22, 'B'] - (df.at[22, 'B'] - df.at[29, 'B']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[21, 'B'] = np.ceil(((df.at[22, 'C'] - df.at[22, 'B']) * 0.382 + df.at[22, 'B'] + df.at[25, 'B']) / 0.05) * 0.05



    df.at[2, 'H'] = np.ceil((1.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'H'] = np.floor(((df.at[2, 'H'] - df.at[2, 'G']) * -0.382 + df.at[2, 'H']) / 0.05) * 0.05
    df.at[6, 'H'] = df4.iloc[7, 7]
    df.at[7, 'H'] = df4.iloc[8, 7]
    df.at[0, 'A'] = pd.to_datetime(df.at[0, 'A'])

    df.at[10, 'H'] = df1[df1['Date'] == df1['Date'].max()]['Low'].values[0]

    df.at[0, 'A'] = df.at[0, 'A'].strftime('%d-%b-%Y')
    df.at[13, 'H'] = np.floor((df.at[7, 'A'] - 1.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'H'] = np.ceil(((df.at[13, 'G'] - df.at[13, 'H']) * 0.382 + df.at[13, 'H']) / 0.05) * 0.05
    df.at[20, 'H'] = np.ceil((1.618 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[21, 'H'] = np.floor(
        (df.at[20, 'H'] - (df.at[20, 'H'] - df.at[20, 'G']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[23, 'H'] = df5.iloc[8, 8]
    df.at[24, 'H'] = df5.iloc[9, 8]
    df.at[25, 'H'] = df5.iloc[10, 8]
    df.at[31, 'H'] = np.floor((df.at[25, 'A'] - 1.618 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[30, 'H'] = np.ceil(
        (df.at[31, 'H'] + (df.at[31, 'G'] - df.at[31, 'H']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05
    df.at[1, 'G'] = np.ceil(((df.at[2, 'H'] - df.at[2, 'G']) * 0.382 + df.at[2, 'G']) / 0.05) * 0.05
    df.at[14, 'G'] = np.floor(((df.at[13, 'G'] - df.at[13, 'H']) * -0.382 + df.at[13, 'G']) / 0.05) * 0.05
    df.at[19, 'G'] = np.ceil(
        ((df.at[20, 'H'] - df.at[20, 'G']) * 0.382 + df.at[20, 'G'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[32, 'G'] = np.floor(
        (df.at[31, 'G'] - (df.at[31, 'G'] - df.at[31, 'H']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[2, 'I'] = np.ceil((2.618 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'I'] = np.floor(((df.at[2, 'I'] - df.at[2, 'H']) * -0.382 + df.at[2, 'I']) / 0.05) * 0.05
    df.at[6, 'I'] = df4.iloc[7, 8]
    df.at[7, 'I'] = df4.iloc[8, 8]
    df.at[8, 'I'] = df4.iloc[9, 8]
    df.at[0, 'A'] = pd.to_datetime(df.at[0, 'A'])

    df.at[10, 'I'] = df1.loc[df1['Date'] == df1['Date'].max(), 'Close'].values[0]


    df.at[0, 'A'] = df.at[0, 'A'].strftime('%d-%b-%Y')
    df.at[13, 'I'] = np.floor((df.at[7, 'A'] - 2.618 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'I'] = np.ceil(((df.at[13, 'H'] - df.at[13, 'I']) * 0.382 + df.at[13, 'I']) / 0.05) * 0.05
    df.at[20, 'I'] = np.ceil((2.618 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[21, 'I'] = np.floor(
        (df.at[20, 'I'] - (df.at[20, 'I'] - df.at[20, 'H']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    df.at[23, 'I'] = df5.iloc[8, 9]
    df.at[24, 'I'] = df5.iloc[9, 9]
    df.at[25, 'I'] = df5.iloc[10, 9]
    df.at[26, 'I'] = df5.iloc[11, 9]
    df.at[31, 'I'] = np.floor((df.at[25, 'A'] - 2.618 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[30, 'I'] = np.ceil(
        (df.at[31, 'I'] + (df.at[31, 'H'] - df.at[31, 'I']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05

    df.at[1, 'H'] = np.ceil(((df.at[2, 'I'] - df.at[2, 'H']) * 0.382 + df.at[2, 'H']) / 0.05) * 0.05
    df.at[14, 'H'] = np.floor(((df.at[13, 'H'] - df.at[13, 'I']) * -0.382 + df.at[13, 'H']) / 0.05) * 0.05
    df.at[19, 'H'] = np.ceil(
        ((df.at[20, 'I'] - df.at[20, 'H']) * 0.382 + df.at[20, 'H'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[32, 'H'] = np.floor(
        (df.at[31, 'H'] - (df.at[31, 'H'] - df.at[31, 'I']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[18, 'C'] = pd.to_datetime(df.at[18, 'C'])

    if df.at[34, 'C'] >= df.at[21, 'B'] and df.at[34, 'C'] <= df.at[21, 'D'] and df.at[34, 'D'] >= df.at[30, 'D'] and \
            df.at[34, 'D'] <= df.at[30, 'B']:
        df.at[34, 'G'] = 'ERROR'
    elif df.at[34, 'C'] >= df.at[21, 'D'] and df.at[34, 'C'] <= df.at[20, 'E']:
        df.at[34, 'G'] = 'RMSL HIGH@RC38.2%'
    elif df.at[34, 'C'] >= df.at[21, 'D'] and df.at[34, 'C'] <= df.at[20, 'F']:
        df.at[34, 'G'] = 'RMSLHIGH@RC61.8%'
    elif df.at[34, 'C'] >= df.at[20, 'E'] and df.at[34, 'C'] <= df.at[19, 'G']:
        df.at[34, 'G'] = 'RMSLHIGH@RC100%'
    elif df.at[34, 'C'] >= df.at[20, 'F'] and df.at[34, 'C'] <= df.at[19, 'H']:
        df.at[34, 'G'] = 'RMSL HIGH@RC127.2%'
    elif df.at[34, 'C'] >= df.at[21, 'B'] and df.at[34, 'C'] <= df.at[21, 'D']:
        df.at[34, 'G'] = 'ICRC'
    elif df.at[34, 'D'] <= df.at[30, 'B'] and df.at[34, 'D'] >= df.at[30, 'D']:
        df.at[34, 'G'] = 'ICFC'
    elif df.at[34, 'D'] <= df.at[30, 'D'] and df.at[34, 'D'] >= df.at[31, 'F']:
        df.at[34, 'G'] = 'RMSLLOW@FC38.2%'
    elif df.at[34, 'D'] <= df.at[30, 'D'] and df.at[34, 'D'] >= df.at[31, 'F']:
        df.at[34, 'G'] = 'RMSLLOW@FC61.8%'
    elif df.at[34, 'D'] <= df.at[31, 'E'] and df.at[34, 'D'] >= df.at[32, 'G']:
        df.at[34, 'G'] = 'RMSLLOW@FC100%'
    elif df.at[34, 'D'] <= df.at[31, 'F'] and df.at[34, 'D'] >= df.at[32, 'H']:
        df.at[34, 'G'] = 'RMSLLOW@FC127.2%'
    elif df.at[34, 'C'] <= df.at[21, 'B'] and df.at[34, 'C'] >= df.at[30, 'B'] and df.at[34, 'D'] >= df.at[30, 'B'] and \
            df.at[34, 'D'] <= df.at[21, 'B']:
        df.at[34, 'G'] = 'ICRR'
    else:
        df.at[34, 'G'] = ''

    matching_rows = df2.loc[
        (df2['to_date'] == df.at[18, 'C'].strftime('%d-%b-%Y')) |
        (df2['from_date'] == df.at[18, 'C'].strftime('%d-%b-%Y'))
        ]

    if not matching_rows.empty:
        df.at[22, 'A'] = matching_rows['High'].values[0]
        df.at[23, 'A'] = matching_rows['Low'].values[0]
    else:
        df.at[22, 'A'] = 0
        df.at[23, 'A'] = 0

    df.at[25, 'A'] = df5.iloc[9, 0]
    df.at[26, 'A'] = df5.iloc[10, 0]
    df.at[18, 'C'] = df.at[18, 'C'].strftime('%d-%b-%y')

    df.at[2, 'J'] = np.ceil((4.236 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'J'] = np.floor(((df.at[2, 'J'] - df.at[2, 'I']) * -0.382 + df.at[2, 'J']) / 0.05) * 0.05
    try:
        df.at[7, 'J'] = np.ceil(((df.at[8, 'A'] * 0.146) + df.at[7, 'I']) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass
    df.at[10, 'J'] = df.at[10, 'G'] - df.at[10, 'H']
    df.at[13, 'J'] = np.floor((df.at[7, 'A'] - 4.236 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'J'] = np.ceil(((df.at[13, 'I'] - df.at[13, 'J']) * 0.382 + df.at[13, 'J']) / 0.05) * 0.05

    df.at[20, 'J'] = np.ceil((4.236 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[21, 'J'] = np.floor(
        (df.at[20, 'J'] - (df.at[20, 'J'] - df.at[20, 'I']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05
    if df.at[25, 'I'] < df.at[25, 'G']:
        df.at[24, 'J'] = 'Bdp Error'
    else:
        df.at[24, 'J'] = ''
    try:
        df.at[25, 'J'] = np.ceil(((df.at[26, 'A'] * 0.146) + df.at[25, 'I']) / 0.1) * 0.1
    except numpy.core._exceptions._UFuncNoLoopError:
        pass
    df.at[31, 'J'] = np.floor((df.at[25, 'A'] - 4.236 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[30, 'J'] = np.ceil(
        (df.at[31, 'J'] + (df.at[31, 'I'] - df.at[31, 'J']) * 0.382 + df.at[25, 'B']) / 0.05) * 0.05
    df.at[1, 'I'] = np.ceil(((df.at[2, 'J'] - df.at[2, 'I']) * 0.382 + df.at[2, 'I']) / 0.05) * 0.05
    df.at[14, 'I'] = np.floor(((df.at[13, 'I'] - df.at[13, 'J']) * -0.382 + df.at[13, 'I']) / 0.05) * 0.05
    df.at[19, 'I'] = np.ceil(
        ((df.at[20, 'J'] - df.at[20, 'I']) * 0.382 + df.at[20, 'I'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[32, 'I'] = np.floor(
        (df.at[31, 'I'] - (df.at[31, 'I'] - df.at[31, 'J']) * 0.382 - df.at[25, 'B']) / 0.05) * 0.05

    df.at[2, 'K'] = np.ceil((6.85 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[3, 'K'] = np.floor(((df.at[2, 'K'] - df.at[2, 'J']) * -0.382 + df.at[2, 'K']) / 0.05) * 0.05
    df.at[6, 'K'] = np.ceil((10.086 * df.at[8, 'A'] + df.at[7, 'A']) / 0.1) * 0.1
    df.at[13, 'K'] = np.floor((df.at[7, 'A'] - 6.85 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[12, 'K'] = np.ceil(((df.at[13, 'J'] - df.at[13, 'K']) * 0.382 + df.at[13, 'K']) / 0.05) * 0.05
    df.at[16, 'K'] = np.floor((df.at[7, 'A'] - 10.086 * df.at[8, 'A']) / 0.1) * 0.1
    df.at[20, 'K'] = np.ceil((6.85 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[21, 'K'] = np.floor(((df.at[20, 'K'] - df.at[20, 'J']) * -0.382 + df.at[20, 'K']) / 0.05) * 0.05
    df.at[24, 'K'] = np.ceil((10.086 * df.at[26, 'A'] + df.at[25, 'A']) / 0.1) * 0.1
    df.at[31, 'K'] = np.floor((df.at[25, 'A'] - 6.85 * df.at[26, 'A']) / 0.1) * 0.1
    df.at[30, 'K'] = np.ceil(((df.at[31, 'J'] - df.at[31, 'K']) * 0.382 + df.at[31, 'K']) / 0.05) * 0.05
    df.at[1, 'K'] = np.ceil(((df.at[6, 'K'] - df.at[2, 'K']) * 0.382 + df.at[2, 'K']) / 0.05) * 0.05
    df.at[14, 'K'] = np.floor(((df.at[13, 'K'] - df.at[16, 'K']) * -0.382 + df.at[13, 'K']) / 0.05) * 0.05
    df.at[19, 'K'] = np.ceil(((df.at[24, 'K'] - df.at[20, 'K']) * 0.382 + df.at[20, 'K']) / 0.05) * 0.05
    df.at[1, 'J'] = np.ceil(((df.at[2, 'K'] - df.at[2, 'J']) * 0.382 + df.at[2, 'J']) / 0.05) * 0.05
    df.at[14, 'J'] = np.floor(((df.at[13, 'J'] - df.at[13, 'K']) * -0.382 + df.at[13, 'J']) / 0.05) * 0.05
    df.at[19, 'J'] = np.ceil(
        ((df.at[20, 'K'] - df.at[20, 'J']) * 0.382 + df.at[20, 'J'] + df.at[25, 'B']) / 0.05) * 0.05
    df.at[32, 'J'] = np.floor(((df.at[31, 'J'] - df.at[31, 'K']) * -0.382 + df.at[31, 'J']) / 0.05) * 0.05
    df.at[34, 'K'] = np.floor((df.at[25, 'A'] - 10.086 * df.at[26, 'A']) / 0.1) * 0.1

    df.at[32, 'K'] = np.floor(((df.at[31, 'K'] - df.at[34, 'K']) * -0.382 + df.at[31, 'K']) / 0.05) * 0.05

    df.at[36, 'A'] = 'Monthly'
    df.at[36, 'B'] = 'Plan For:'
    df.at[36, 'C'] = df7.iloc[1, 3]
    df.at[36, 'D'] = df7.iloc[1, 5]

    df.at[36, 'C'] = pd.to_datetime(df.at[36, 'C'])

    high_values = df3[(df3['to'] == df.at[36, 'C'].strftime('%d-%b-%Y')) | (df3['from'] == df.at[36, 'C'].strftime('%d-%b-%Y'))]['HIGH'].values
    if len(high_values) > 0:
        df.at[40, 'A'] = high_values[0]
    else:
        df.at[40, 'A'] = 0


    low_values = df3[(df3['to'] <= df.at[36, 'C'].strftime('%d-%b-%Y')) & (df3['from'] >= df.at[36, 'C'].strftime('%d-%b-%Y'))]['LOW'].values
    if len(low_values) > 0:
        df.at[41, 'A'] = low_values[0]
    else:
        df.at[41, 'A'] = 0


    df.at[36, 'C'] = df.at[36, 'C'].strftime('%d-%b-%Y')
    df.at[43, 'A'] = df7.iloc[9, 0]
    df.at[44, 'A'] = df7.iloc[10, 0]
    df.at[42, 'A'] = np.ceil((df.at[44, 'A'] * 0.046 + df.at[43, 'A']) / 0.1) * 0.1
    df.at[45, 'A'] = np.floor((df.at[43, 'A'] - df.at[44, 'A'] * 0.046) / 0.1) * 0.1

    df.at[36, 'C'] = pd.to_datetime(df.at[36, 'C'])
    date1 = df.at[36, 'C'] - pd.Timedelta(days=1)
    date_values = df1[(df1['Date'] <= date1.strftime('%d-%b-%Y'))]['Date'].values
    if len(date_values) > 0:
        df.at[51, 'A'] = date_values[-1]
    else:
        df.at[51, 'A'] = 0

    df.at[51, 'A'] = pd.to_datetime(df.at[51, 'A'])
    df.at[51, 'A'] = df.at[51, 'A'].strftime('%d-%b-%y')

    df.at[52, 'A'] = df.at[36, 'C']

    df.at[40, 'B'] = np.ceil((0.146 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[43, 'B'] = np.ceil((df.at[43, 'A'] * 0.00073) / 0.05) * 0.05
    df.at[44, 'B'] = np.ceil((df.at[44, 'A'] * 0.073 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[47, 'B'] = np.floor((df.at[43, 'A'] - 0.146 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[41, 'B'] = np.floor(
        (df.at[40, 'B'] - (df.at[40, 'B'] - df.at[47, 'B']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[46, 'B'] = np.ceil(
        ((df.at[40, 'B'] - df.at[47, 'B']) * 0.382 + df.at[47, 'B'] + df.at[43, 'B']) / 0.05) * 0.05


    df.at[40, 'C'] = np.ceil((0.236 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[43, 'C'] = 'TREND : '
    df.at[44, 'C'] = 'LAG : '
    df.at[47, 'C'] = np.floor((df.at[43, 'A'] - 0.236 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[39, 'B'] = np.ceil(
        ((df.at[40, 'C'] - df.at[40, 'B']) * 0.382 + df.at[40, 'B'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[48, 'B'] = np.floor(
        (df.at[47, 'B'] - (df.at[47, 'B'] - df.at[47, 'C']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05

    df.at[40, 'D'] = np.ceil((0.382 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[41, 'D'] = np.floor(
        (df.at[40, 'D'] - (df.at[40, 'D'] - df.at[40, 'C']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[47, 'D'] = np.floor((df.at[43, 'A'] - 0.382 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[46, 'D'] = np.ceil(
        (df.at[47, 'D'] + (df.at[47, 'C'] - df.at[47, 'D']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05

    df.at[52, 'D'] = ''

    df.at[39, 'E'] = np.ceil((0.618 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[48, 'E'] = np.floor((df.at[43, 'A'] - 0.618 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[47, 'E'] = np.ceil(
        (df.at[48, 'E'] + (df.at[47, 'D'] - df.at[48, 'E']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[39, 'D'] = np.ceil(
        ((df.at[39, 'E'] - df.at[40, 'D']) * 0.382 + df.at[40, 'D'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[48, 'D'] = np.floor(
        (df.at[47, 'D'] - (df.at[47, 'D'] - df.at[48, 'E']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05

    df.at[52, 'E'] = ''

    df.at[39, 'F'] = np.ceil((1.0 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[40, 'F'] = np.floor(
        (df.at[39, 'F'] - (df.at[39, 'F'] - df.at[39, 'E']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[48, 'F'] = np.floor((df.at[43, 'A'] - 1.0 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[47, 'F'] = np.ceil(
        (df.at[48, 'F'] + (df.at[48, 'E'] - df.at[48, 'F']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[38, 'E'] = np.ceil(
        ((df.at[39, 'F'] - df.at[39, 'E']) * 0.382 + df.at[39, 'E'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[49, 'E'] = np.floor(
        (df.at[48, 'E'] - (df.at[48, 'E'] - df.at[48, 'F']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[52, 'F'] = 'Open Tone:'

    df.at[38, 'G'] = np.ceil((1.272 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[39, 'G'] = np.floor(
        (df.at[38, 'G'] - (df.at[38, 'G'] - df.at[39, 'F']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[41, 'G'] = 'MSF'
    df.at[43, 'G'] = df7.iloc[10, 6]
    df.at[44, 'G'] = df7.iloc[11, 6]
    try:
        df.at[45, 'G'] = np.floor((df.at[44, 'G'] - df.at[44, 'A'] * 0.146) / 0.1) * 0.1
    except TypeError:
        pass
    df.at[49, 'G'] = np.floor((df.at[43, 'A'] - 1.272 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[38, 'F'] = np.ceil(
        ((df.at[38, 'G'] - df.at[39, 'F']) * 0.382 + df.at[39, 'F'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[49, 'F'] = np.floor(
        (df.at[48, 'F'] - (df.at[48, 'F'] - df.at[49, 'G']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05

    df.at[38, 'H'] = np.ceil((1.618 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[39, 'H'] = np.floor(
        (df.at[38, 'H'] - (df.at[38, 'H'] - df.at[38, 'G']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[41, 'H'] = df7.iloc[8, 7]
    df.at[42, 'H'] = df7.iloc[9, 7]
    df.at[43, 'H'] = df7.iloc[10, 7]
    df.at[49, 'H'] = np.floor((df.at[43, 'A'] - 1.618 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[48, 'H'] = np.ceil(
        (df.at[49, 'H'] + (df.at[49, 'G'] - df.at[49, 'H']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[37, 'G'] = np.ceil(
        ((df.at[38, 'H'] - df.at[38, 'G']) * 0.382 + df.at[38, 'G'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[50, 'G'] = np.floor(
        (df.at[49, 'G'] - (df.at[49, 'G'] - df.at[49, 'H']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05

    df.at[38, 'I'] = np.ceil((2.618 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[39, 'I'] = np.floor(
        (df.at[38, 'I'] - (df.at[38, 'I'] - df.at[38, 'H']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    df.at[41, 'I'] = df7.iloc[8, 8]
    df.at[42, 'I'] = df7.iloc[9, 8]
    df.at[43, 'I'] = df7.iloc[10, 8]
    df.at[44, 'I'] = df7.iloc[11, 8]
    df.at[49, 'I'] = np.floor((df.at[43, 'A'] - 2.618 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[48, 'I'] = np.ceil(
        (df.at[49, 'I'] + (df.at[49, 'H'] - df.at[49, 'I']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[37, 'H'] = np.ceil(
        ((df.at[38, 'I'] - df.at[38, 'H']) * 0.382 + df.at[38, 'H'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[50, 'H'] = np.floor((df.at[49, 'H'] - (df.at[49, 'H'] - df.at[49, 'I']) * 0.382 - df.at[43, 'B']))

    df.at[38, 'J'] = np.ceil((4.236 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[39, 'J'] = np.floor(
        (df.at[38, 'J'] - (df.at[38, 'J'] - df.at[38, 'I']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05
    if df.at[43, 'I'] < df.at[43, 'G']:
        df.at[42, 'J'] = "Bdp error"
    else:
        df.at[42, 'J'] = ''
    try:
        df.at[43, 'J'] = np.ceil(((df.at[44, 'A'] * 0.146) + df.at[43, 'I']) / 0.1) * 0.1
    except TypeError:
        pass
    df.at[49, 'J'] = np.floor((df.at[43, 'A'] - 4.236 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[48, 'J'] = np.ceil(
        (df.at[49, 'J'] + (df.at[49, 'I'] - df.at[49, 'J']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[37, 'I'] = np.ceil(
        ((df.at[38, 'J'] - df.at[38, 'I']) * 0.382 + df.at[38, 'I'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[50, 'I'] = np.floor(
        (df.at[49, 'I'] - (df.at[49, 'I'] - df.at[49, 'J']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05

    df.at[38, 'K'] = np.ceil((6.85 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[39, 'K'] = np.floor(((df.at[38, 'K'] - df.at[38, 'J']) * -0.382 + df.at[38, 'K']) / 0.05) * 0.05
    df.at[42, 'K'] = np.ceil((10.086 * df.at[44, 'A'] + df.at[43, 'A']) / 0.1) * 0.1
    df.at[49, 'K'] = np.floor((df.at[43, 'A'] - 6.85 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[48, 'K'] = np.ceil(((df.at[49, 'J'] - df.at[49, 'K']) * 0.382 + df.at[49, 'K']) / 0.05) * 0.05
    df.at[37, 'K'] = np.ceil(((df.at[42, 'K'] - df.at[38, 'K']) * 0.382 + df.at[38, 'K']) / 0.05) * 0.05
    df.at[52, 'K'] = np.floor((df.at[43, 'A'] - 10.086 * df.at[44, 'A']) / 0.1) * 0.1
    df.at[37, 'J'] = np.ceil(
        ((df.at[38, 'K'] - df.at[38, 'J']) * 0.382 + df.at[38, 'J'] + df.at[43, 'B']) / 0.05) * 0.05
    df.at[50, 'J'] = np.floor(((df.at[49, 'J'] - df.at[49, 'K']) * -0.382 + df.at[49, 'J']) / 0.05) * 0.05
    df.at[50, 'K'] = np.floor(((df.at[49, 'K'] - df.at[52, 'K']) * -0.382 + df.at[49, 'K']) / 0.05) * 0.05

    df.at[48, 'G'] = np.ceil(
        (df.at[49, 'G'] + (df.at[48, 'F'] - df.at[49, 'G']) * 0.382 + df.at[43, 'B']) / 0.05) * 0.05
    df.at[40, 'E'] = np.floor(
        (df.at[39, 'E'] - (df.at[39, 'E'] - df.at[40, 'D']) * 0.382 - df.at[43, 'B']) / 0.05) * 0.05

    df.at[36, 'C'] = df.at[36, 'C'].strftime('%d-%b-%y')
    df.at[51, 'A'] = pd.to_datetime(df.at[51, 'A'])
    df.at[51, 'A'] = df.at[51, 'A'].strftime('%d-%b-%y')
    df.at[52, 'A'] = pd.to_datetime(df.at[52, 'A'])
    df.at[52, 'A'] = df.at[52, 'A'].strftime('%d-%b-%y')

    df.at[51, 'A'] = pd.to_datetime(df.at[51, 'A'])
    a51 = df1.loc[df1['Date'] == df.at[51, 'A'], 'Date'].values
    third_date = 0
    if len(a51) > 0:
        date_to_find = a51[0]
        index_of_date = df1[df1['Date'] == date_to_find].index[0]
        if index_of_date + 3 < len(df1):
            third_date = df1.iloc[index_of_date + 3]['Date']
        else:
            pass


    third_date = pd.to_datetime(third_date)
    df.at[51, 'B'] = third_date
    try:
        df.at[51, 'B'] = df.at[51, 'B'].strftime('%d-%b-%y')
    except ValueError:
        pass
    df.at[51, 'A'] = df.at[51, 'A'].strftime('%d-%b-%y')

    df.at[52, 'A'] = pd.to_datetime(df.at[52, 'A'])
    df.at[51, 'B'] = pd.to_datetime(df.at[51, 'B'])
    days_number1 = 7 - (df.at[51, 'B'].dayofweek) - 1
    df.at[52, 'B'] = df.at[51, 'B'] + pd.DateOffset(days=days_number1)
    try:
        df.at[51, 'B'] = df.at[51, 'B'].strftime('%d-%b-%y')
    except ValueError:
        pass
    df.at[52, 'A'] = df.at[52, 'A'].strftime('%d-%b-%y')

    try:
        df.at[52, 'B'] = df.at[52, 'B'].strftime('%d-%b-%y')
    except ValueError:
        pass

    df.at[24, 'A'] = df.at[42, 'A']
    df.at[27, 'A'] = df.at[45, 'A']

    df.at[0, 'L'] = ''

    df.at[1, 'M'] = 'Time'
    df.at[5, 'M'] = 'T.Phase:'
    df.at[6, 'M'] = 'PCS:'
    df.at[7, 'M'] = 'PCR:'
    df.at[8, 'M'] = 'Indicators'

    df.at[19, 'M'] = 'POR UP:'
    df.at[20, 'M'] = 'Date'
    df.at[27, 'M'] = 'T.Phase:'
    df.at[28, 'M'] = 'PCR:'
    df.at[29, 'M'] = 'Indicators :'
    df.at[36, 'M'] = 'POR UP:'
    df.at[37, 'M'] = 'Date'
    df.at[44, 'M'] = 'T.Phase'
    df.at[45, 'M'] = 'PCR:'
    df.at[46, 'M'] = 'Indicators : '

    df.at[1, 'N'] = 'Resistance'
    try:
        if df.at[7, 'I'] <= df.at[11, 'B'] and df.at[7, 'I'] >= df.at[11, 'D']:
            df.at[8, 'N'] = 'BSB'
        else:
            df.at[8, 'N'] = ''
        if df.at[8, 'G'] <= df.at[4, 'D'] and df.at[8, 'G'] >= df.at[4, 'B']:
            df.at[9, 'N'] = 'LG'
        else:
            df.at[9, 'N'] = ''
        if df.at[8, 'G'] >= df.at[12, 'E'] and df.at[8, 'G'] <= df.at[11, 'D']:
            df.at[10, 'N'] = 'CCP.FC'
        else:
            df.at[10, 'N'] = ''
        if df.at[8, 'G'] >= df.at[12, 'F'] and df.at[8, 'G'] <= df.at[12, 'E']:
            df.at[11, 'N'] = 'CATCH.KNIFE'
        else:
            df.at[11, 'N'] = ''
        if df.at[7, 'I'] <= df.at[4, 'D'] and df.at[7, 'I'] >= df.at[4, 'B']:
            df.at[12, 'N'] = 'KK.BUY'
        else:
            df.at[12, 'N'] = ''
        if np.isnan(df.at[23, 'A']):
            df.at[19, 'N'] = df.at[26, 'A']
        else:
            df.at[19, 'N'] = df.at[23, 'A'] + df.at[26, 'A']
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[20, 'N'] = 'Resistance'
    df.at[27, 'N'] = 6574
    df.at[28, 'N'] = 6473


    try:
        if df.at[25, 'I'] <= df.at[29, 'B'] and df.at[25, 'I'] >= df.at[29, 'D']:
            df.at[29, 'N'] = 'BSB'
        else:
            df.at[29, 'N'] = ''
        if df.at[26, 'G'] <= df.at[22, 'D'] and df.at[26, 'G'] >= df.at[22, 'B']:
            df.at[30, 'N'] = 'LG'
        else:
            df.at[30, 'N'] = ''
        if df.at[26, 'G'] >= df.at[30, 'E'] and df.at[26, 'G'] <= df.at[29, 'D']:
            df.at[31, 'N'] = 'CCP.FC'
        else:
            df.at[31, 'N'] = ''
        if df.at[26, 'G'] >= df.at[30, 'F'] and df.at[26, 'G'] <= df.at[30, 'E']:
            df.at[32, 'N'] = 'CATCH.KNIFE'
        else:
            df.at[32, 'N'] = ''
        if df.at[25, 'I'] <= df.at[22, 'D'] and df.at[25, 'I'] >= df.at[22, 'B']:
            df.at[33, 'N'] = 'KK.BUY'
        else:
            df.at[33, 'N'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[36, 'N'] = df.at[41, 'A'] + df.at[44, 'A']
    df.at[37, 'N'] = 'Resistance'
    try:
        if df.at[43, 'I'] <= df.at[47, 'B'] and df.at[43, 'I'] >= df.at[47, 'D']:
            df.at[46, 'N'] = 'BSB'
        else:
            df.at[46, 'N'] = ''
        if df.at[44, 'G'] <= df.at[40, 'D'] and df.at[44, 'G'] >= df.at[40, 'B']:
            df.at[47, 'N'] = 'LG'
        else:
            df.at[47, 'N'] = ''
        if df.at[44, 'G'] >= df.at[48, 'E'] and df.at[44, 'G'] <= df.at[47, 'D']:
            df.at[48, 'N'] = 'CCP.FC'
        else:
            df.at[48, 'N'] = ''
        if df.at[44, 'G'] >= df.at[48, 'F'] and df.at[44, 'G'] <= df.at[48, 'E']:
            df.at[49, 'N'] = 'CATCH.KNIFE'
        else:
            df.at[49, 'N'] = ''
        if df.at[43, 'I'] <= df.at[40, 'D'] and df.at[43, 'I'] >= df.at[40, 'B']:
            df.at[50, 'N'] = 'KK.BUY'
        else:
            df.at[50, 'N'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[6, 'O'] = 'Support'

    try:
        if df.at[7, 'I'] <= df.at[11, 'B'] and df.at[7, 'I'] >= df.at[12, 'D']:
            df.at[8, 'O'] = 'HG'
        else:
            df.at[8, 'O'] = ''
        if df.at[8, 'G'] <= df.at[4, 'D'] and df.at[8, 'G'] >= df.at[4, 'B']:
            df.at[9, 'O'] = 'RSB'
        else:
            df.at[9, 'O'] = ''

        if df.at[7, 'I'] <= df.at[3, 'E'] and df.at[7, 'I'] >= df.at[4, 'D']:
            df.at[10, 'O'] = 'CCP.RC'
        else:
            df.at[10, 'O'] = ''
        if df.at[7, 'I'] <= df.at[3, 'F'] and df.at[7, 'I'] >= df.at[3, 'E']:
            df.at[11, 'O'] = 'CUT.KITE'
        else:
            df.at[11, 'O'] = ''
        if df.at[8, 'G'] >= df.at[11, 'D'] and df.at[8, 'G'] <= df.at[11, 'B']:
            df.at[12, 'O'] = 'KK.SELL'
        else:
            df.at[12, 'O'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[19, 'O'] = 'POR DOWN :'
    df.at[26, 'O'] = 6577
    df.at[27, 'O'] = 'Support'
    df.at[28, 'O'] = 'PCS:'

    try:
        if df.at[25, 'I'] <= df.at[29, 'B'] and df.at[25, 'I'] >= df.at[29, 'D']:
            df.at[29, 'O'] = 'HG'
        else:
            df.at[29, 'O'] = ''
        if df.at[26, 'G'] <= df.at[22, 'D'] and df.at[26, 'G'] >= df.at[22, 'B']:
            df.at[30, 'O'] = 'RSB'
        else:
            df.at[30, 'O'] = ''
        if df.at[25, 'I'] <= df.at[21, 'E'] and df.at[25, 'I'] >= df.at[22, 'D']:
            df.at[31, 'O'] = 'CCP.RC'
        else:
            df.at[31, 'O'] = ''
        if df.at[25, 'I'] <= df.at[21, 'F'] and df.at[25, 'I'] >= df.at[21, 'E']:
            df.at[32, 'O'] = 'CUT.KITE'
        else:
            df.at[32, 'O'] = ''
        if df.at[26, 'G'] >= df.at[29, 'D'] and df.at[26, 'G'] <= df.at[29, 'B']:
            df.at[33, 'O'] = 'KK.SELL'
        else:
            df.at[33, 'O'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[36, 'O'] = 'POR DOWN:'
    df.at[44, 'O'] = 'Support'
    df.at[45, 'O'] = 'PCS:'

    try:
        if df.at[43, 'I'] <= df.at[47, 'B'] and df.at[43, 'I'] >= df.at[47, 'D']:
            df.at[46, 'O'] = 'HG'
        else:
            df.at[46, 'O'] = ''
        if df.at[44, 'G'] <= df.at[40, 'D'] and df.at[44, 'G'] >= df.at[40, 'B']:
            df.at[47, 'O'] = 'RSB'
        else:
            df.at[47, 'O'] = ''
        if df.at[43, 'I'] <= df.at[39, 'E'] and df.at[43, 'I'] >= df.at[40, 'D']:
            df.at[48, 'O'] = 'CCP.RC'
        else:
            df.at[48, 'O'] = ''
        if df.at[43, 'I'] <= df.at[39, 'F'] and df.at[43, 'I'] >= df.at[39, 'E']:
            df.at[49, 'O'] = 'CUT.KITE'
        else:
            df.at[49, 'O'] = ''
        if df.at[44, 'G'] <= df.at[47, 'D'] and df.at[44, 'G'] >= df.at[47, 'B']:
            df.at[50, 'O'] = 'KK.SELL'
        else:
            df.at[50, 'O'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass

    df.at[1, 'P'] = 'D.P.'
    df.at[3, 'P'] = 'TREND'
    df.at[4, 'P'] = 'LAG'
    df.at[5, 'P'] = 'OPEN'
    df.at[6, 'P'] = 'POR'
    df.at[7, 'P'] = 'S/R'
    df.at[8, 'P'] = 'C.B/S'
    df.at[9, 'P'] = 'RC.23.6%'
    df.at[10, 'P'] = 'FC.23.6%'
    if np.isnan(df.at[22, 'A']):
        df.at[19, 'P'] = df.at[26, 'A']
    else:
        df.at[19, 'P'] = abs(df.at[22, 'A'] - df.at[26, 'A'])
    df.at[27, 'P'] = 'Date'
    df.at[28, 'P'] = 6454
    df.at[36, 'P'] = df.at[40, 'A'] - df.at[44, 'A']
    df.at[44, 'P'] = 'Date'
    df.at[45, 'P'] = 6137.9

    df.at[0, 'R'] = 'QTRLY'
    df.at[20, 'R'] = 'DATE'
    df.at[51, 'R'] = df.at[0, 'A']

    df.at[1, 'R'] = df8.iloc[6, 7]
    df.at[2, 'R'] = df8.iloc[6, 8]
    df.at[3, 'R'] = df8.iloc[6, 3]
    df.at[4, 'R'] = df8.iloc[7, 3]
    df.at[5, 'R'] = df8.iloc[16, 6]
    if df.at[4, 'R'] == "uptrend continues" or df.at[4, 'R'] == "pullback buy starts" or df.at[4, 'R'] == "trend reversal buy" or df.at[4, 'R'] == "cor comp org uptrend begins":
        df.at[6, 'R'] = df8.at[0, 'N']
    else:
        df.at[6, 'R'] = df8.at[0, 'P']
    df.at[7, 'R'] = df8.iloc[4, 8]
    df.at[8, 'R'] = df8.iloc[5, 8]
    df.at[9, 'R'] = df8.iloc[4, 2]
    df.at[10, 'R'] = df8.iloc[11, 2]
    df.at[11, 'R'] = 'LVL TRADED'
    df.at[12, 'R'] = 'S.BULL'
    df.at[13, 'R'] = 'S.BEAR'
    df.at[14, 'R'] = 'BDP'
    df.at[15, 'R'] = 'WDP'
    df.at[16, 'R'] = 'JGD(3+1)'
    df.at[17, 'R'] = 'JWD(2+2)'
    df.at[18, 'R'] = 'MSF'

    df.at[11, 'S'] = 'MON'

    df.at[0, 'T'] = 'HLF.YRLY'
    df.at[1, 'T'] = df8.iloc[23, 7]
    df.at[2, 'T'] = df8.iloc[23, 8]
    df.at[3, 'T'] = df8.iloc[23, 3]
    df.at[4, 'T'] = df8.iloc[25, 3]
    df.at[5, 'T'] = df8.iloc[33, 6]
    if df.at[4, 'T'] == "uptrend continues" or df.at[4, 'T'] == "pullback buy starts" or df.at[4, 'T'] == "trend reversal buy" or df.at[4, 'T'] == "cor comp org uptrend begins":
        df.at[6, 'T'] = df8.iloc[17, 11]
    else:
        df.at[6, 'T'] = df8.iloc[17, 13]
    df.at[7, 'T'] = df8.iloc[21, 8]
    df.at[8, 'T'] = df8.iloc[22, 8]
    df.at[9, 'T'] = df8.iloc[21, 2]
    df.at[10, 'T'] = df8.iloc[28, 2]
    df.at[11, 'T'] = 'TUE'

    df.at[11, 'U'] = 'WED'

    df.at[0, 'V'] = 'YRLY'
    df.at[1, 'V'] = df8.iloc[41, 7]
    df.at[2, 'V'] = df8.iloc[41, 8]
    df.at[3, 'V'] = df8.iloc[41, 3]
    df.at[4, 'V'] = df8.iloc[42, 4]
    df.at[5, 'V'] = df8.iloc[51, 6]
    if df.at[4, 'V'] == "uptrend continues" or df.at[4, 'V'] == "pullback buy starts" or df.at[4, 'V'] == "trend reversal buy" or df.at[4, 'V'] == "cor comp org uptrend begins":
        df.at[6, 'V'] = df8.iloc[35, 11]
    else:
        df.at[6, 'V'] = df8.iloc[35, 13]
    df.at[7, 'V'] = df8.iloc[39, 8]
    df.at[8, 'V'] = df8.iloc[40, 8]
    df.at[9, 'V'] = df8.iloc[39, 2]
    df.at[10, 'V'] = df8.iloc[46, 2]
    df.at[11, 'V'] = 'THU'

    df.at[11, 'W'] = 'FRI'

    df.at[20, 'R'] = 'DATE'
    df.at[51, 'R'] = df.at[0, 'A']
    df.at[51, 'R'] = pd.to_datetime(df.at[51, 'R'])


    df1['Date'] = pd.to_datetime(df1['Date'])
    end_date = df.at[51, 'R']
    found_dates = []
    current_date = end_date - pd.DateOffset(days=1)

    while len(found_dates) < 30:
        if current_date in df1['Date'].values:
            found_dates.append(current_date)
        current_date -= pd.DateOffset(days=1)

    found_dates = found_dates[::-1]

    start_row = 21
    end_row = 50

    if len(found_dates) <= (end_row - start_row + 1):
        for element in found_dates:
            df.loc[start_row, 'R'] = element
            start_row += 1

    df.at[20, 'S'] = 'HIGH'
    df.at[20, 'T'] = 'LOW'
    df.at[20, 'U'] = 'CLOSE'
    df.at[20, 'V'] = 'RANGE'
    df.at[20, 'W'] = 'MSF'

    for i in range(21, 52):
        df.at[i, 'R'] = pd.to_datetime(df.at[i, 'R'])
        is_high = df1.loc[df1['Date'] == df.at[i, 'R'].strftime('%d-%b-%Y'), 'High'].values
        if len(is_high) > 0:
            df.at[i, 'S'] = is_high[0]
        else:
            pass

        ti_low = df1.loc[df1['Date'] == df.at[i, 'R'].strftime('%d-%b-%Y'), 'Low'].values
        if len(ti_low) > 0:
            df.at[i, 'T'] = ti_low[0]
        else:
            pass

        ui_close = df1.loc[df1['Date'] == df.at[i, 'R'].strftime('%d-%b-%Y'), 'Close'].values
        if len(ui_close) > 0:
            df.at[i, 'U'] = ui_close[0]
        else:
            pass

        vi_range = df1.loc[df1['Date'] == df.at[i, 'R'].strftime('%d-%b-%Y'), 'Range'].values
        if len(vi_range) > 0:
            df.at[i, 'V'] = vi_range[0]
        else:
            pass

        wi_msf = df1.loc[df1['Date'] == df.at[i, 'R'].strftime('%d-%b-%Y'), 'MSF'].values
        if len(wi_msf) > 0:
            df.at[i, 'W'] = wi_msf[0]
        else:
            pass

    df.at[0, 'A'] = pd.to_datetime(df.at[0, 'A'])
    df1['Date'] = pd.to_datetime(df1['Date'])



    df.at[0, 'X'] = df.at[0, 'A']
    df.at[2, 'X'] = 'PDH'
    df.at[3, 'X'] = 'PDL'
    df.at[4, 'X'] = 'PDC'
    df.at[5, 'X'] = 'RANGE'
    df.at[6, 'X'] = 'E6'
    df.at[7, 'X'] = 'CP'
    df.at[8, 'X'] = 'E30'
    df.at[9, 'X'] = 'Mod.. CP'
    df.at[10, 'X'] = 'UT1'
    df.at[11, 'X'] = 'UT2'
    df.at[12, 'X'] = 'UT3'
    df.at[14, 'X'] = 'LT1'
    df.at[15, 'X'] = 'LT2'
    df.at[16, 'X'] = 'LT3'
    df.at[19, 'X'] = df.at[18, 'C']
    df.at[20, 'X'] = 'PWH'
    df.at[21, 'X'] = 'PWL'
    df.at[22, 'X'] = 'PWC'
    df.at[23, 'X'] = 'RANGE'
    df.at[24, 'X'] = 'E6'
    df.at[25, 'X'] = 'CP'
    df.at[26, 'X'] = 'E30'
    df.at[27, 'X'] = 'MISS. CP'
    df.at[28, 'X'] = 'UT1'
    df.at[29, 'X'] = 'UT2'
    df.at[30, 'X'] = 'UT3'
    df.at[32, 'X'] = 'LT1'
    df.at[33, 'X'] = 'LT2'
    df.at[34, 'X'] = 'LT3'
    df.at[37, 'X'] = df.at[36, 'C']
    df.at[38, 'X'] = 'PMH'
    df.at[39, 'X'] = 'PML'
    df.at[40, 'X'] = 'PMC'
    df.at[41, 'X'] = 'E6'
    df.at[42, 'X'] = 'CP'
    df.at[43, 'X'] = 'MISS. CP'
    df.at[44, 'X'] = 'RANGE'
    df.at[45, 'X'] = 'UT1'
    df.at[46, 'X'] = 'UT2'
    df.at[47, 'X'] = 'UT3'
    df.at[49, 'X'] = 'LT1'
    df.at[50, 'X'] = 'LT2'
    df.at[51, 'X'] = 'LT3'

    df.at[0, 'Y'] = 'Daily'
    df.at[0, 'X'] = pd.to_datetime(df.at[0, 'X'])
    date_x = df.at[0, 'X'] - pd.Timedelta(days=1)

    df.at[2, 'Y'] = df1.loc[df1['Date'] == df1['Date'].max(), 'High'].values
    df.at[3, 'Y'] = df1.loc[df1['Date'] == df1['Date'].max(), 'Low'].values
    df.at[4, 'Y'] = df1.loc[df1['Date'] == df1['Date'].max(), 'Close'].values


    df.at[5, 'Y'] = df.at[2, 'Y'] - df.at[3, 'Y']

    df.at[6, 'Y'] = df1.loc[df1['Date'] == df1['Date'].max(), 'E6'].values
    df.at[7, 'Y'] = df1.loc[df1['Date'] == df1['Date'].max(), 'CP'].values
    df.at[8, 'Y'] = df1.loc[df1['Date'] == df1['Date'].max(), 'E30'].values

    df.at[10, 'Y'] = df.at[4, 'Y'] + (df.at[5, 'Y'] / 2)
    df.at[11, 'Y'] = df.at[4, 'Y'] + (df.at[5, 'Y'])
    df.at[12, 'Y'] = df.at[4, 'Y'] + (df.at[5, 'Y'] * 1.5)
    df.at[14, 'Y'] = df.at[4, 'Y'] - (df.at[5, 'Y'] / 2)
    df.at[15, 'Y'] = df.at[4, 'Y'] - (df.at[5, 'Y'])
    df.at[16, 'Y'] = df.at[4, 'Y'] - (df.at[5, 'Y'] * 1.5)
    df.at[18, 'Y'] = 'Weekly'
    df.at[19, 'Y'] = df.at[18, 'D']

    df.at[19, 'X'] = pd.to_datetime(df.at[19, 'X'])
    date_x2 = df.at[19, 'X'] - pd.Timedelta(days=1)
    df.at[20, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'High'].values[0]
    df.at[21, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'Low'].values[0]
    df.at[22, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'Close'].values[0]
    df.at[23, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'RANGE'].values[0]
    df.at[24, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'E6'].values[0]
    df.at[25, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'CP'].values[0]
    df.at[26, 'Y'] = df2.loc[(df2['to_date'] == date_x2.strftime('%d-%b-%Y')) | (
                df2['from_date'] == date_x2.strftime('%d-%b-%Y')), 'E30'].values[0]
    df.at[28, 'Y'] = df.at[22, 'Y'] + (df.at[23, 'Y'] / 2)
    df.at[29, 'Y'] = df.at[22, 'Y'] + (df.at[23, 'Y'])
    df.at[30, 'Y'] = df.at[22, 'Y'] + (df.at[23, 'Y'] * 1.5)
    df.at[32, 'Y'] = df.at[22, 'Y'] - (df.at[23, 'Y'] / 2)
    df.at[33, 'Y'] = df.at[22, 'Y'] - (df.at[23, 'Y'])
    df.at[34, 'Y'] = df.at[22, 'Y'] - (df.at[23, 'Y'] * 1.5)
    df.at[36, 'Y'] = 'Monthly'
    df.at[37, 'Y'] = df.at[36, 'D']
    df.at[37, 'X'] = pd.to_datetime(df.at[37, 'X'])
    date_x3 = df.at[37, 'X'] - pd.Timedelta(days=1)
    df.at[38, 'Y'] = \
    df3.loc[(df3['to'] <= date_x3.strftime('%d-%b-%Y')) & (df3['from'] >= date_x3.strftime('%d-%b-%Y')), 'HIGH'].values[
        0]
    df.at[39, 'Y'] = \
    df3.loc[(df3['to'] <= date_x3.strftime('%d-%b-%Y')) & (df3['from'] >= date_x3.strftime('%d-%b-%Y')), 'LOW'].values[
        0]
    df.at[40, 'Y'] = df3.loc[
        (df3['to'] <= date_x3.strftime('%d-%b-%Y')) & (df3['from'] >= date_x3.strftime('%d-%b-%Y')), 'CLOSE'].values[0]
    df.at[41, 'Y'] = \
    df3.loc[(df3['to'] <= date_x3.strftime('%d-%b-%Y')) & (df3['from'] >= date_x3.strftime('%d-%b-%Y')), 'E6'].values[0]
    df.at[42, 'Y'] = \
    df3.loc[(df3['to'] <= date_x3.strftime('%d-%b-%Y')) & (df3['from'] >= date_x3.strftime('%d-%b-%Y')), 'CP'].values[0]
    df.at[44, 'Y'] = df3.loc[
        (df3['to'] <= date_x3.strftime('%d-%b-%Y')) & (df3['from'] >= date_x3.strftime('%d-%b-%Y')), 'RANGE'].values[0]
    df.at[45, 'Y'] = df.at[40, 'Y'] + (df.at[44, 'Y'] / 2)
    df.at[46, 'Y'] = df.at[40, 'Y'] + (df.at[44, 'Y'])
    df.at[47, 'Y'] = df.at[40, 'Y'] + (df.at[44, 'Y'] * 1.5)
    df.at[49, 'Y'] = df.at[40, 'Y'] - (df.at[44, 'Y'] / 2)
    df.at[50, 'Y'] = df.at[40, 'Y'] - (df.at[44, 'Y'])
    df.at[51, 'Y'] = df.at[40, 'Y'] - (df.at[44, 'Y'] * 1.5)
    df.at[37, 'X'] = df.at[37, 'X'].strftime('%d-%b-%y')

    df.at[0, 'AA'] = 'DAILY'
    df.at[1, 'AA'] = 'BDP'
    df.at[2, 'AA'] = 'WDP'
    df.at[3, 'AA'] = 'JGD'
    df.at[4, 'AA'] = 'JWD'
    df.at[5, 'AA'] = 'S.BULL'
    df.at[6, 'AA'] = 'S.BEAR'
    df.at[7, 'AA'] = 'PCS'
    df.at[8, 'AA'] = 'PCR'
    df.at[9, 'AA'] = 'T.PHASE'
    df.at[10, 'AA'] = 'MSF'

    if df.at[25, 'D'] == "UPTREND CONTINUES" or df.at[25, 'D'] == "PULLBACK BUY STARTS" or df.at[
        25, 'D'] == "TREND REVERSAL BUY" or df.at[25, 'D'] == "COR COMP ORG UPTREND BEGINS":
        df.at[14, 'AA'] = "UPTREND"
    elif df.at[25, 'D'] == "DOWNTREND CONTINUES" or df.at[25, 'D'] == "CORRECTION SELL STARTS" or df.at[
        25, 'D'] == "TREND REVERSAL SELL" or df.at[25, 'D'] == "PULLBK COMP ORG DOWNTREND BEGINS":
        df.at[14, 'AA'] = "DOWNTREND"
    else:
        df.at[14, 'AA'] = ''

    df.at[18, 'AA'] = 'WEEKLY'
    df.at[19, 'AA'] = 'BDP'
    df.at[20, 'AA'] = 'WDP'
    df.at[21, 'AA'] = 'JWD'
    df.at[22, 'AA'] = 'JGD'
    df.at[23, 'AA'] = 'MSF'
    df.at[24, 'AA'] = 'PCR'
    df.at[25, 'AA'] = 'PCS'
    df.at[26, 'AA'] = 'T.PHASE'
    df.at[27, 'AA'] = 'POR.UP'
    df.at[28, 'AA'] = 'POR.DN'
    df.at[35, 'AA'] = 'MONTHLY'
    df.at[36, 'AA'] = 'BDP'
    df.at[37, 'AA'] = 'WDP'
    df.at[38, 'AA'] = 'JWD'
    df.at[39, 'AA'] = 'JGD'
    df.at[40, 'AA'] = 'MSF'
    df.at[41, 'AA'] = 'PCR'
    df.at[42, 'AA'] = 'PCS'
    df.at[43, 'AA'] = 'T.PHASE'
    df.at[44, 'AA'] = 'POR.UP'
    df.at[45, 'AA'] = 'POR.DN'

    df.at[0, 'AB'] = ''
    df.at[1, 'AB'] = df.at[7, 'I']
    df.at[2, 'AB'] = df.at[8, 'G']
    df.at[3, 'AB'] = df.at[7, 'G']
    df.at[4, 'AB'] = df.at[8, 'I']
    df.at[5, 'AB'] = df.at[6, 'A']
    df.at[6, 'AB'] = df.at[9, 'A']
    df.at[7, 'AB'] = df.at[6, 'N']
    df.at[8, 'AB'] = df.at[7, 'N']
    df.at[9, 'AB'] = df.at[5, 'N']
    df.at[10, 'AB'] = df.at[6, 'H']
    df.at[19, 'AB'] = df.at[25, 'I']
    df.at[20, 'AB'] = df.at[26, 'G']
    df.at[21, 'AB'] = df.at[26, 'I']
    df.at[22, 'AB'] = df.at[25, 'G']
    df.at[23, 'AB'] = df.at[23, 'H']
    df.at[24, 'AB'] = df.at[28, 'N']
    df.at[25, 'AB'] = df.at[28, 'P']
    df.at[26, 'AB'] = df.at[27, 'N']
    df.at[27, 'AB'] = df.at[19, 'N']
    df.at[28, 'AB'] = df.at[19, 'P']
    df.at[29, 'AB'] = df.at[24, 'A']
    df.at[30, 'AB'] = df.at[27, 'A']
    df.at[36, 'AB'] = df.at[43, 'I']
    df.at[37, 'AB'] = df.at[44, 'G']
    df.at[38, 'AB'] = df.at[44, 'I']
    df.at[39, 'AB'] = df.at[43, 'G']
    df.at[40, 'AB'] = df.at[41, 'H']
    df.at[41, 'AB'] = df.at[45, 'N']
    df.at[42, 'AB'] = df.at[45, 'P']
    df.at[43, 'AB'] = df.at[44, 'N']
    df.at[44, 'AB'] = df.at[36, 'N']
    df.at[45, 'AB'] = df.at[36, 'P']

    df.at[0, 'AC'] = ''
    df.at[2, 'AC'] = 'HIGH'
    df.at[3, 'AC'] = 'S/R'
    df.at[4, 'AC'] = 'C.B/S'
    df.at[5, 'AC'] = 'RC.23.6%'
    df.at[6, 'AC'] = 'FC.23.6%'
    df.at[8, 'AC'] = 'LOW'
    df.at[9, 'AC'] = 'S/R'
    df.at[10, 'AC'] = 'C.B/S'
    df.at[11, 'AC'] = 'RC.23.6%'
    df.at[12, 'AC'] = 'FC.23.6%'

    df.at[0, 'AD'] = ''
    df.at[2, 'AD'] = 'QTR'
    df.at[0, 'AE'] = ''
    df.at[0, 'AF'] = ''

    df.at[8, 'AH'] = 'DATE'



    df.at[6, 'A'] = np.ceil((df.at[26, 'A'] * 0.073 + df.at[25, 'A']) / 0.1) * 0.1
    df.at[9, 'A'] = np.floor((df.at[25, 'A'] - df.at[26, 'A'] * 0.073) / 0.1) * 0.1

    df.at[5, 'AB'] = df.at[6, 'A']
    df.at[6, 'AB'] = df.at[9, 'A']

    desired_date = df.at[0, 'A'] - pd.Timedelta(days=1)

    found_date = None
    while found_date is None:
        matching_dates = df1.loc[df1['Date'] == desired_date, 'Date'].values
        if len(matching_dates) > 0:
            found_date = matching_dates[0]
        else:
            desired_date -= pd.Timedelta(days=1)
    if found_date is not None:
        df.at[11, 'AH'] = found_date



    desired_date2 = df.at[11, 'AH'] - pd.Timedelta(days=1)

    found_date = None
    while found_date is None:
        matching_dates = df1.loc[df1['Date'] == desired_date2, 'Date'].values
        if len(matching_dates) > 0:
            found_date = matching_dates[0]
        else:
            desired_date2 -= pd.Timedelta(days=1)
    if found_date is not None:
        df.at[10, 'AH'] = found_date



    df.at[12, 'AH'] = df.at[0, 'A'].strftime('%d-%b-%y')

    df.at[10, 'AH'] = pd.to_datetime(df.at[10, 'AH'])
    df.at[10, 'AH'] = df.at[10, 'AH'].strftime('%d-%b-%y')
    df.at[11, 'AH'] = pd.to_datetime(df.at[11, 'AH'])
    df.at[11, 'AH'] = df.at[11, 'AH'].strftime('%d-%b-%y')
    df.at[12, 'AH'] = pd.to_datetime(df.at[12, 'AH'])
    df.at[12, 'AH'] = df.at[0, 'A'].strftime('%d-%b-%y')
    df.at[19, 'X'] = df.at[19, 'X'].strftime('%d-%b-%y')
    df.at[0, 'X'] = df.at[0, 'X'].strftime('%d-%b-%y')

    df.at[8, 'AI'] = 'HIGH'
    df.at[10, 'AI'] = df1.loc[df1['Date'] == df.at[10, 'AH'], 'High'].values[0]
    df.at[11, 'AI'] = df1.loc[df1['Date'] == df.at[11, 'AH'], 'High'].values[0]


    df.at[8, 'AJ'] = 'LOW'
    df.at[10, 'AJ'] = df1.loc[df1['Date'] == df.at[10, 'AH'], 'Low'].values[0]
    df.at[11, 'AJ'] = df1.loc[df1['Date'] == df.at[11, 'AH'], 'Low'].values[0]

    df.at[8, 'AK'] = 'LOW'
    df.at[10, 'AK'] = df1.loc[df1['Date'] == df.at[10, 'AH'], 'Close'].values[0]
    df.at[11, 'AK'] = df1.loc[df1['Date'] == df.at[11, 'AH'], 'Close'].values[0]


    df.at[12, 'AI'] = df.at[11, 'AI']       # because the last date is the next monday date, so the values are not present for that date so
    df.at[12, 'AJ'] = df.at[11, 'AJ']       # taking the values of the last available date.
    df.at[12, 'AK'] = df.at[11, 'AK']


    df.at[8, 'AL'] = 'HIGH'
    df.at[10, 'AL'] = max([df.at[10, 'AI'], df.at[11, 'AI']])
    df.at[8, 'AM'] = 'LOW'
    df.at[10, 'AM'] = min([df.at[10, 'AJ'], df.at[11, 'AJ']])

    df.at[20, 'AD'] = 1
    df.at[21, 'AD'] = 2
    df.at[22, 'AD'] = 3
    df.at[23, 'AD'] = 4
    df.at[24, 'AD'] = 5
    df.at[25, 'AD'] = 6
    df.at[26, 'AD'] = 7
    df.at[27, 'AD'] = 8
    df.at[3, 'AD'] = (df.at[7, 'R'] - df.at[10, 'AL']) / df.at[7, 'R'] * 100
    df.at[4, 'AD'] = (df.at[8, 'R'] - df.at[10, 'AL']) / df.at[8, 'R'] * 100
    df.at[5, 'AD'] = (df.at[9, 'R'] - df.at[10, 'AL']) / df.at[9, 'R'] * 100
    df.at[6, 'AD'] = (df.at[10, 'R'] - df.at[10, 'AL']) / df.at[10, 'R'] * 100
    df.at[8, 'AD'] = 'QTR'
    df.at[9, 'AD'] = (df.at[10, 'AM'] - df.at[7, 'R']) / df.at[10, 'AM'] * 100
    df.at[10, 'AD'] = (df.at[10, 'AM'] - df.at[8, 'R']) / df.at[10, 'AM'] * 100
    df.at[11, 'AD'] = (df.at[10, 'AM'] - df.at[9, 'R']) / df.at[10, 'AM'] * 100
    df.at[12, 'AD'] = (df.at[10, 'AM'] - df.at[10, 'R']) / df.at[10, 'AM'] * 100

    df.at[2, 'AE'] = 'HALF.YR'
    df.at[3, 'AE'] = (df.at[7, 'T'] - df.at[10, 'AL']) / df.at[7, 'T'] * 100
    df.at[4, 'AE'] = (df.at[8, 'T'] - df.at[10, 'AL']) / df.at[8, 'T'] * 100
    df.at[5, 'AE'] = (df.at[9, 'T'] - df.at[10, 'AL']) / df.at[9, 'T'] * 100
    df.at[6, 'AE'] = (df.at[10, 'T'] - df.at[10, 'AL']) / df.at[10, 'T'] * 100
    df.at[8, 'AE'] = 'HALF.YR'
    df.at[9, 'AE'] = (df.at[10, 'AM'] - df.at[7, 'T']) / df.at[10, 'AM'] * 100
    df.at[10, 'AE'] = (df.at[10, 'AM'] - df.at[8, 'T']) / df.at[10, 'AM'] * 100
    df.at[11, 'AE'] = (df.at[10, 'AM'] - df.at[9, 'T']) / df.at[10, 'AM'] * 100
    df.at[12, 'AE'] = (df.at[10, 'AM'] - df.at[10, 'T']) / df.at[10, 'AM'] * 100
    df.at[20, 'AE'] = 'UPTREND CONTINUES'
    df.at[21, 'AE'] = 'PULLBACK BUY STARTS'
    df.at[22, 'AE'] = 'TREND REVERSAL BUY'
    df.at[23, 'AE'] = 'COR COMP ORG UPTREND BEGINS'
    df.at[24, 'AE'] = 'DOWNTREND CONTINUES'
    df.at[25, 'AE'] = 'CORRECTION SELL STARTS'
    df.at[26, 'AE'] = 'TREND REVERSAL SELL'
    df.at[27, 'AE'] = 'PULLBK COMP ORG DOWNTREND BEGINS'

    df.at[2, 'AF'] = 'YR'
    try:
        df.at[3, 'AF'] = (df.at[7, 'V'] - df.at[10, 'AL']) / df.at[7, 'V'] * 100
    except RuntimeWarning:
        pass
    try:
        df.at[4, 'AF'] = (df.at[8, 'V'] - df.at[10, 'AL']) / df.at[8, 'V'] * 100
    except RuntimeWarning:
        pass
    try:
        df.at[5, 'AF'] = (df.at[9, 'V'] - df.at[10, 'AL']) / df.at[9, 'V'] * 100
    except RuntimeWarning:
        pass
    try:
        df.at[6, 'AF'] = (df.at[10, 'V'] - df.at[10, 'AL']) / df.at[10, 'V'] * 100
    except RuntimeWarning:
        pass
    df.at[8, 'AF'] = 'YR'
    try:
        df.at[9, 'AF'] = (df.at[10, 'AM'] - df.at[7, 'V']) / df.at[10, 'AM'] * 100
    except RuntimeWarning:
        pass
    try:
        df.at[10, 'AF'] = (df.at[10, 'AM'] - df.at[8, 'V']) / df.at[10, 'AM'] * 100
    except RuntimeWarning:
        pass
    try:
        df.at[11, 'AF'] = (df.at[10, 'AM'] - df.at[9, 'V']) / df.at[10, 'AM'] * 100
    except RuntimeWarning:
        pass
    try:
        df.at[12, 'AF'] = (df.at[10, 'AM'] - df.at[10, 'V']) / df.at[10, 'AM'] * 100
    except RuntimeWarning:
        pass


    if (df.at[5, 'AD'] <= 1.5 and df.at[5, 'AD'] >= -1.5) or (df.at[11, 'AD'] <= 1.5 and df.at[11, 'AD'] >= -1.5):
        df.at[13, 'M'] = "MARKET NEAR QUARTERLY RC 23.6%"
    elif (df.at[6, 'AD'] <= 1.5 and df.at[6, 'AD'] >= -1.5) or (df.at[12, 'AD'] <= 1.5 and df.at[12, 'AD'] >= -1.5):
        df.at[13, 'M'] = "MARKET NEAR QUARTERLY FC 23.6%"
    else:
        df.at[13, 'M'] = ''

    if (df.at[5, 'AE'] <= 1.5 and df.at[5, 'AE'] >= -1.5) or (df.at[11, 'AE'] <= 1.5 and df.at[11, 'AE'] >= -1.5):
        df.at[14, 'M'] = "MARKET NEAR HALF YEARLY RC 23.6%"
    elif (df.at[6, 'AE'] <= 1.5 and df.at[6, 'AE'] >= -1.5) or (df.at[12, 'AE'] <= 1.5 and df.at[12, 'AE'] >= -1.5):
        df.at[14, 'M'] = "MARKET NEAR HALF YEARLY FC 23.6%"
    else:
        df.at[14, 'M'] = ''

    if (df.at[5, 'AF'] <= 1.5 and df.at[5, 'AF'] >= -1.5) or (df.at[11, 'AF'] <= 1.5 and df.at[11, 'AF'] >= -1.5):
        df.at[15, 'M'] = "MARKET NEAR YEARLY RC 23.6%"
    elif (df.at[6, 'AF'] <= 1.5 and df.at[6, 'AF'] >= -1.5) or (df.at[12, 'AF'] <= 1.5 and df.at[12, 'AF'] >= -1.5):
        df.at[15, 'M'] = "MARKET NEAR YEARLY FC 23.6%"
    else:
        df.at[15, 'M'] = ''

    if (df.at[3, 'AD'] <= 1.5 and df.at[3, 'AD'] >= 0) or (df.at[9, 'AD'] <= 1.5 and df.at[9, 'AD'] >= 0):
        df.at[16, 'M'] = "Market Near QTRLY S/R"
    elif (df.at[4, 'AD'] <= 1.5 and df.at[4, 'AD'] >= 0) or (df.at[10, 'AD'] <= 1.5 and df.at[10, 'AD'] >= 0):
        df.at[16, 'M'] = "Market Near QTRLY C.B/S"
    else:
        df.at[16, 'M'] = ''

    if (df.at[3, 'AE'] <= 1.5 and df.at[3, 'AE'] >= 0) or (df.at[9, 'AE'] <= 1.5 and df.at[9, 'AE'] >= 0):
        df.at[17, 'M'] = "MARKET NEAR HLF.YRLY S/R"
    elif (df.at[4, 'AE'] <= 1.5 and df.at[4, 'AE'] >= 0) or (df.at[10, 'AE'] <= 1.5 and df.at[10, 'AE'] >= 0):
        df.at[17, 'M'] = "MARKET NEAR HLF.YRLY C.B/S"
    else:
        df.at[17, 'M'] = ''

    if (df.at[3, 'AF'] <= 1.5 and df.at[3, 'AF'] >= 0) or (df.at[9, 'AF'] <= 1.5 and df.at[9, 'AF'] >= 0):
        df.at[18, 'M'] = "MARKET NEAR YEARLY S/R"
    elif (df.at[4, 'AF'] <= 2 and df.at[4, 'AF'] >= 0) or (df.at[10, 'AF'] <= 2 and df.at[10, 'AF'] >= 0):
        df.at[18, 'M'] = "MARKET NEAR YEARLY C.B/S"
    else:
        df.at[18, 'M'] = ''

    drange_list = []



    drange_list.append(
        [df.at[1, 'AB'], df.at[2, 'AB'], df.at[3, 'AB'], df.at[4, 'AB'], df.at[5, 'AB'], df.at[6, 'AB'], df.at[7, 'AB'],
         df.at[8, 'AB'], df.at[9, 'AB'], df.at[10, 'AB']])



    df['AB'] = df['AB'].replace({'': None})


    list1_key = [df.at[1, 'AA'], df.at[2, 'AA'], df.at[3, 'AA'], df.at[4, 'AA'], df.at[5, 'AA'], df.at[6, 'AA'],
                 df.at[7, 'AA'], df.at[8, 'AA'], df.at[9, 'AA'], df.at[10, 'AA']]
    list1_value = [df.at[1, 'AB'], df.at[2, 'AB'], df.at[3, 'AB'], df.at[4, 'AB'], df.at[5, 'AB'], df.at[6, 'AB'],
                   df.at[7, 'AB'], df.at[8, 'AB'], df.at[9, 'AB'], df.at[10, 'AB']]

    list2_key = [df.at[19, 'AA'], df.at[20, 'AA'], df.at[21, 'AA'], df.at[22, 'AA'], df.at[23, 'AA'], df.at[24, 'AA'],
                 df.at[25, 'AA'], df.at[26, 'AA'], df.at[27, 'AA'], df.at[28, 'AA']]
    list2_value = [df.at[19, 'AB'], df.at[20, 'AB'], df.at[21, 'AB'], df.at[22, 'AB'], df.at[23, 'AB'], df.at[24, 'AB'],
                   df.at[25, 'AB'], df.at[26, 'AB'], df.at[27, 'AB'], df.at[28, 'AB']]

    list3_key = [df.at[36, 'AA'], df.at[37, 'AA'], df.at[38, 'AA'], df.at[39, 'AA'], df.at[40, 'AA'], df.at[41, 'AA'],
                 df.at[42, 'AA'], df.at[43, 'AA'], df.at[44, 'AA'], df.at[45, 'AA']]
    list3_value = [df.at[36, 'AB'], df.at[37, 'AB'], df.at[38, 'AB'], df.at[39, 'AB'], df.at[40, 'AB'], df.at[41, 'AB'],
                   df.at[42, 'AB'], df.at[43, 'AB'], df.at[44, 'AB'], df.at[45, 'AB']]

    def drange_func2(val1, val2, list_drange_key, list_drange_value):
        closest_values = []

        if list_drange_value and list_drange_value[0] is not None:
            min_difference = abs(val1 - list_drange_value[0]) + abs(val2 - list_drange_value[0])
            closest_values = [list_drange_key[0]]

            for i in range(1, len(list_drange_value)):
                item = list_drange_value[i]
                if item is not None:
                    diff = abs(val1 - item) + abs(val2 - item)
                    if diff < min_difference:
                        min_difference = diff
                        closest_values = [list_drange_key[i]]
                    elif diff == min_difference:
                        closest_values.append(list_drange_key[i])

        return closest_values

    df.at[1, 'A'] = drange_func2(df.at[11, 'B'], df.at[4, 'B'], list1_key, list1_value)
    df.at[14, 'A'] = drange_func2(df.at[11, 'B'], df.at[4, 'B'], list1_key, list1_value)
    df.at[19, 'A'] = drange_func2(df.at[29, 'B'], df.at[22, 'B'], list2_key, list2_value)
    df.at[51, 'G'] = drange_func2(df.at[49, 'H'], df.at[49, 'G'], list3_key, list3_value)
    df.at[33, 'G'] = drange_func2(df.at[31, 'H'], df.at[31, 'G'], list1_key, list1_value)
    df.at[32, 'D'] = drange_func2(df.at[30, 'E'], df.at[29, 'D'], list2_key, list2_value)
    df.at[14, 'F'] = drange_func2(df.at[13, 'G'], df.at[12, 'F'], list1_key, list1_value)
    df.at[14, 'D'] = drange_func2(df.at[12, 'E'], df.at[11, 'D'], list1_key, list1_value)
    df.at[18, 'E'] = drange_func2(df.at[21, 'E'], df.at[21, 'F'], list2_key, list2_value)
    df.at[32, 'A'] = drange_func2(df.at[29, 'B'], df.at[22, 'B'], list2_key, list2_value)
    df.at[51, 'E'] = drange_func2(df.at[48, 'F'], df.at[48, 'E'], list3_key, list3_value)
    df.at[2, 'B'] = drange_func2(df.at[4, 'B'], df.at[4, 'D'], list1_key, list1_value)
    df.at[13, 'C'] = drange_func2(df.at[11, 'D'], df.at[11, 'B'], list1_key, list1_value)
    df.at[20, 'C'] = drange_func2(df.at[22, 'B'], df.at[22, 'D'], list2_key, list2_value)
    df.at[31, 'C'] = drange_func2(df.at[29, 'D'], df.at[29, 'B'], list2_key, list2_value)
    df.at[19, 'D'] = drange_func2(df.at[22, 'D'], df.at[21, 'E'], list2_key, list2_value)
    df.at[15, 'E'] = drange_func2(df.at[12, 'F'], df.at[12, 'E'], list1_key, list1_value)
    df.at[19, 'F'] = drange_func2(df.at[21, 'F'], df.at[20, 'G'], list2_key, list2_value)
    df.at[33, 'F'] = drange_func2(df.at[30, 'F'], df.at[30, 'E'], list2_key, list2_value)
    df.at[15, 'G'] = drange_func2(df.at[13, 'H'], df.at[13, 'G'], list1_key, list1_value)
    df.at[18, 'G'] = drange_func2(df.at[20, 'G'], df.at[20, 'H'], list2_key, list2_value)
    df.at[33, 'I'] = drange_func2(df.at[31, 'I'], df.at[31, 'H'], list2_key, list2_value)
    df.at[15, 'J'] = drange_func2(df.at[13, 'J'], df.at[13, 'I'], list1_key, list1_value)
    df.at[37, 'A'] = drange_func2(df.at[47, 'B'], df.at[40, 'B'], list3_key, list3_value)
    df.at[50, 'A'] = drange_func2(df.at[47, 'B'], df.at[40, 'B'], list3_key, list3_value)
    df.at[38, 'C'] = drange_func2(df.at[40, 'B'], df.at[40, 'D'], list3_key, list3_value)
    df.at[49, 'C'] = drange_func2(df.at[47, 'D'], df.at[47, 'B'], list3_key, list3_value)
    df.at[37, 'D'] = drange_func2(df.at[40, 'D'], df.at[39, 'E'], list3_key, list3_value)
    df.at[50, 'D'] = drange_func2(df.at[48, 'E'], df.at[47, 'D'], list3_key, list3_value)
    df.at[51, 'F'] = drange_func2(df.at[48, 'F'], df.at[48, 'E'], list3_key, list3_value)
    df.at[37, 'F'] = drange_func2(df.at[39, 'F'], df.at[38, 'G'], list3_key, list3_value)
    df.at[36, 'G'] = drange_func2(df.at[38, 'G'], df.at[38, 'H'], list3_key, list3_value)
    df.at[36, 'I'] = drange_func2(df.at[38, 'H'], df.at[38, 'I'], list3_key, list3_value)
    df.at[0, 'E'] = drange_func2(df.at[3, 'E'], df.at[3, 'F'], list1_key, list1_value)
    df.at[36, 'E'] = drange_func2(df.at[39, 'E'], df.at[39, 'F'], list3_key, list3_value)
    df.at[1, 'D'] = drange_func2(df.at[4, 'D'], df.at[3, 'E'], list1_key, list1_value)
    df.at[18, 'I'] = drange_func2(df.at[20, 'H'], df.at[20, 'I'], list2_key, list2_value)
    df.at[0, 'G'] = drange_func2(df.at[2, 'G'], df.at[2, 'H'], list1_key, list1_value)
    df.at[52, 'A'] = pd.to_datetime(df.at[52, 'A'])
    df.at[52, 'B'] = pd.to_datetime(df.at[52, 'B'])
    try:
        start_date = df.at[52, 'A'].strftime('%d-%b-%Y')
        end_date = df.at[52, 'B'].strftime('%d-%b-%Y')
        high3 = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'High']
        low3 = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'Low']
        close3 = df1.loc[(df1['Date'] >= start_date) & (df1['Date'] <= end_date), 'Close']




        # because in some cases high3 and low3 are null
        try:
            df.at[52, 'C'] = max(high3)
            df.at[52, 'D'] = min(low3)
        except ValueError:
            df.at[52, 'C'] = ''
            df.at[52, 'D'] = ''

        try:
            df.at[52, 'E'] = close3.iloc[-1]
        except IndexError:
            df.at[52, 'E'] = ''

    except ValueError:
        pass


    try:
        df.at[52, 'A'] = df.at[52, 'A'].strftime('%d-%b-%y')
        df.at[52, 'B'] = df.at[52, 'B'].strftime('%d-%b-%y')
    except ValueError:
        pass


    try:
        if df.at[52, 'C'] >= df.at[39, 'B'] and df.at[52, 'C'] <= df.at[39, 'D'] and df.at[52, 'D'] >= df.at[48, 'D'] and \
                df.at[52, 'D'] <= df.at[48, 'B']:
            df.at[52, 'G'] = 'ERROR'
        elif df.at[52, 'C'] >= df.at[39, 'D'] and df.at[52, 'C'] <= df.at[38, 'E']:
            df.at[52, 'G'] = 'RMSLHIGH@RC38.2%'
        elif df.at[52, 'C'] >= df.at[39, 'D'] and df.at[52, 'C'] <= df.at[38, 'F']:
            df.at[52, 'G'] = 'RMSLHIGH@RC61.8%'
        elif df.at[52, 'C'] >= df.at[38, 'E'] and df.at[52, 'C'] <= df.at[37, 'G']:
            df.at[52, 'G'] = 'RMSLHIGH@RC100%'
        elif df.at[52, 'C'] >= df.at[38, 'F'] and df.at[52, 'C'] <= df.at[37, 'H']:
            df.at[52, 'G'] = 'RMSL HIGH@RC127.2%'
        elif df.at[52, 'C'] >= df.at[39, 'B'] and df.at[52, 'C'] <= df.at[39, 'D']:
            df.at[52, 'G'] = 'ICRC'
        elif df.at[52, 'D'] <= df.at[48, 'B'] and df.at[52, 'D'] >= df.at[48, 'D']:
            df.at[52, 'G'] = 'ICFC'
        elif df.at[52, 'D'] <= df.at[48, 'D'] and df.at[52, 'D'] >= df.at[49, 'F']:
            df.at[52, 'G'] = 'RMSLLOW@FC38.2%'
        elif df.at[52, 'D'] <= df.at[48, 'D'] and df.at[52, 'D'] >= df.at[49, 'F']:
            df.at[52, 'G'] = 'RMSLLOW@FC61.8%'
        elif df.at[52, 'D'] <= df.at[49, 'E'] and df.at[52, 'D'] >= df.at[50, 'G']:
            df.at[52, 'G'] = 'RMSLLOW@FC100%'
        elif df.at[52, 'D'] <= df.at[49, 'F'] and df.at[52, 'D'] >= df.at[50, 'H']:
            df.at[52, 'G'] = 'RMSLLOW@FC127.2%'
        elif df.at[52, 'C'] <= df.at[39, 'B'] and df.at[52, 'C'] >= df.at[48, 'B'] and df.at[52, 'D'] >= df.at[48, 'B'] and \
                df.at[52, 'D'] <= df.at[39, 'B']:
            df.at[52, 'G'] = 'ICRR'
        else:
            df.at[52, 'G'] = ''
    except numpy.core._exceptions._UFuncNoLoopError:
        pass


    for i in range(21, 52):
        df.at[i, 'R'] = df.at[i, 'R'].strftime('%d-%b-%y')

    df.at[12, 'AH'] = pd.to_datetime(df.at[12, 'AH'])
    df.at[12, 'AH'] = df.at[12, 'AH'].strftime('%d-%b-%y')
    df.at[0, 'A'] = df.at[0, 'A'].strftime('%d-%b-%y')

    df.at[0, 'A'] = pd.to_datetime(df.at[0, 'A'])
    df.at[0, 'A'] = df.at[0, 'A'].strftime('%d-%b-%y')
    new_df10 = df.copy()
    name11 = df1.at[1, 'symbol']
    # excel_write2(name11, 'Combined-DWM', new_df10)
    return new_df10



def daily_summary(df1, df2, df8):


    last_index_list = df1[df1['Date'].notnull()].index.tolist()
    last_index = last_index_list[-1]

    new_msf_color = df2['MSF_COLOR'][df2['MSF_COLOR'] != ''].tolist()


    data_dict = {
        'SCRIPT': df1.at[1, 'symbol'],
        'DAILYMSF': df8.at[6, 'I'],
        'Pattern': df8.at[6, 'G'],
        'D_HIGH': df8.at[10, 'G'],
        'D_LOW': df8.at[10, 'H'],
        'D_CLOSE': df8.at[10, 'I'],
        'DR1': df8.at[4, 'B'],
        'DR2': df8.at[4, 'C'],
        'DR3': df8.at[4, 'D'],
        'DR4': df8.at[3, 'E'],
        'DR5': df8.at[3, 'F'],
        'DR6': df8.at[2, 'G'],
        'DF1': df8.at[11, 'B'],
        'DF2': df8.at[11, 'C'],
        'DF3': df8.at[11, 'D'],
        'DF4': df8.at[12, 'E'],
        'DF5': df8.at[12, 'F'],
        'DF6': df8.at[13, 'G'],
        'DMSF': df8.at[10, 'AB'],
        'DBDP': df8.at[1, 'AB'],
        'WEEKLY_MSF_PATTERN': df8.at[24, 'H'],
        'WR1': df8.at[22, 'B'],
        'WR2': df8.at[22, 'C'],
        'WR3': df8.at[22, 'D'],
        'WR4': df8.at[21, 'E'],
        'WR5': df8.at[21, 'F'],
        'WR6': df8.at[20, 'G'],
        'WF1': df8.at[29, 'B'],
        'WF2': df8.at[29, 'C'],
        'WF3': df8.at[29, 'D'],
        'WF4': df8.at[30, 'E'],
        'WF5': df8.at[30, 'F'],
        'WF6': df8.at[31, 'G'],
        'WMSF': df8.at[23, 'AB'],
        'WBDP': df8.at[19, 'AB'],
        'MONTHLY_MSF_PATTERN': df8.at[42, 'H'],
        'MR1': df8.at[40, 'B'],
        'MR2': df8.at[40, 'C'],
        'MR3': df8.at[40, 'D'],
        'MR4': df8.at[39, 'E'],
        'MR5': df8.at[39, 'F'],
        'MR6': df8.at[38, 'G'],
        'MF1': df8.at[47, 'B'],
        'MF2': df8.at[47, 'C'],
        'MF3': df8.at[47, 'D'],
        'MF4': df8.at[48, 'E'],
        'MF5': df8.at[48, 'F'],
        'MF6': df8.at[49, 'G'],
        'M_MSF': df8.at[40, 'AB'],
        'M_BDP': df8.at[36, 'AB'],
        'WEEKLY_COLOR': df8.at[23, 'I'],
        'MONTHLY_COLOR': df8.at[41, 'I'],
        'AJ1': df1['AJ'].iloc[last_index],
        'AK1': df1['AK'].iloc[last_index],
        'AJ2': df1['AJ'].iloc[last_index - 1],
        'AK2': df1['AK'].iloc[last_index - 1],
        'AJ3': df1['AJ'].iloc[last_index - 2],
        'AK3': df1['AK'].iloc[last_index - 2],
        'H2': df8.at[49, 'S'],
        'L2': df8.at[49, 'T'],
        'H3': df8.at[48, 'S'],
        'L3': df8.at[48, 'T'],
        'H4': df8.at[47, 'S'],
        'L4': df8.at[47, 'T'],
        'H5': df8.at[46, 'S'],
        'L5': df8.at[46, 'T'],
        'H6': df8.at[45, 'S'],
        'L6': df8.at[45, 'T'],
        'H7': df8.at[44, 'S'],
        'L7': df8.at[44, 'T'],
        'WMSF1': new_msf_color[-1],
        'WMSF2': new_msf_color[-2],
        'WMSF3': new_msf_color[-3],
        'WMSF3_1': new_msf_color[-4],
        'DAILY_HG': df8.at[8, 'O'],
        'DAILY_LG': df8.at[9, 'N'],
        'WEEKLY_HG': df8.at[29, 'O'],
        'WEEKLY_LG': df8.at[30, 'N'],
        'MONTHLY_HG': df8.at[46, 'O'],
        'MONTHLY_LG': df8.at[47, 'N'],

        'WF5L': df8.at[31, 'F'],
        'WF4H': df8.at[29, 'E'],
        'WF4L': df8.at[31, 'E'],
        'WF1H': df8.at[28, 'B'],

        'WR5H': df8.at[20, 'F'],
        'WR4L': df8.at[22, 'E'],
        'WR4H': df8.at[20, 'E'],
        'WR1L': df8.at[23, 'B'],
        'MF5L': df8.at[49, 'F'],
        'MF4H': df8.at[47, 'E'],
        'MF4L': df8.at[49, 'E'],


        'MF1H': df8.at[46, 'B'],
        'MR5H': df8.at[38, 'F'],
        'MR4L': df8.at[40, 'E'],
        'MR4H': df8.at[38, 'E'],
        'MR1L': df8.at[41, 'B'],

        'DNWDP': df8.at[9, 'G'],
        'DNBDP': df8.at[7, 'J'],
        'WNWDP': df8.at[27, 'G'],
        'WNBDP': df8.at[25, 'J'],
        'WHIGH': df2['High'][::-1].loc[df2['High'][::-1] != 0].iloc[0],
        'WLOW': df2['Low'][::-1].loc[df2['Low'][::-1] != 0].iloc[0],

        'WEEK_FACTOR': df8.at[25, 'B'],
        'MONTH_FACTOR': df8.at[43, 'B'],
        'DATE': df1.at[0, 'symbol'].strftime('%Y-%m-%d'),
    }

    df = pd.DataFrame(data_dict, index=[0])

    new_df = df.copy()
    return new_df



def params_api(start_date, end_date, symbol):

    start_date = pd.to_datetime(start_date).timestamp()
    end_date = pd.to_datetime(end_date).timestamp()

    x = requests.get(
        url=f'https://api.tradeclue.com/v1/tv/history?from={int(start_date)}&to={int(end_date)}&symbol={symbol}&resolution=D'
    )
    z = x.json()
    df = pd.DataFrame(z)
    columns_to_keep = ['symbol', 't', 'h', 'l', 'c']
    df = df[columns_to_keep]
    new_df = df.copy()
    return new_df



def api_url(df):

    df.rename(

        columns={'t': 'Date', 'h': 'High', 'l': 'Low', 'c': 'Close'},
        inplace=True
    )

    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df['Date'] = df['Date'].dt.strftime('%d-%b-%Y')
    last_date = df['Date'].iloc[-1]
    df.at[0, 'Symbol'] = last_date
    df.at[1, 'Symbol'] = df.at[0, 'symbol']

    df.drop(columns=['symbol'], inplace=True)

    new_order = ['Symbol', 'Date', 'High', 'Low', 'Close']
    df = df[new_order]
    new_df2 = df.copy()
    return new_df2



def function_call(data_sheet):
    a1 = daily_calc(data_sheet)
    b1 = weekly_calc(a1)
    c1 = monthly_calc(a1)
    d1 = quaterly_calc(a1)
    e1 = half_yearly_calc(a1)
    f1 = yearly_calc(a1)
    h1 = weekly_plan_calc(a1, b1)
    g1 = daily_plan_calc(a1, h1)
    i1 = monthly_plan_calc(a1, c1)
    j1 = combined_qhy_calc(a1, b1, d1, e1, f1)
    k1 = combined_dwm_calc(a1, b1, c1, g1, h1, i1, j1)
    l1 = daily_summary(a1, b1, k1)
    return l1



def process_symbol(symbol, start_date, end_date):

    result_df = params_api(start_date, end_date, symbol)
    new_result_df = api_url(result_df)
    return new_result_df



@app.get('/inputs/{start_date}/{end_date}/{symbol2}')
async def get_values(start_date: str, end_date: str, symbol2: str):
    symbols = symbol2.split(',')
    partial_func = partial(process_symbol, start_date=start_date, end_date=end_date)

    with multiprocessing.Pool(processes=3) as p1:
        a = p1.map(partial_func, symbols)
        b = p1.map(function_call, a)

    # daily summary creation and directory creation
    end_date_new = pd.to_datetime(end_date)
    final_end_date = end_date_new.strftime('%Y-%b-%d')
    final_df = pd.concat(b, ignore_index=True)

    file_path = write_to_excel(final_df, 'daily_summary', final_end_date)

    final_df.columns = final_df.columns.str.lower()
    return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=f"daily_summary_{final_end_date}.xlsx")




if __name__ == '__main__':
    print(datetime.now())

    # reading all symbols from file - 186 symbols
    # df_symbol = pd.read_csv('C:/Users/user/Downloads/fo_mktlots_1.csv')
    # list_symbols_2 = df_symbol['SYMBOL    '].to_list()
    # list_symbols = list(map(str.strip, list_symbols_2))
    #
    # start_date = '2 feb 2023'
    # end_date = '17 oct 2023'
    # symbol2 = list_symbols



    # console input start date, end date and list of symbols
    start_date = input('Start Date:')
    end_date = input('End Date:')
    symbol2 = input('Symbols:')
    symbol2 = [s.strip() for s in symbol2.split(',')]

    # multi-processing logic
    partial_func = partial(process_symbol, start_date=start_date, end_date=end_date)

    with multiprocessing.Pool(processes=3) as p1:
        a = p1.map(partial_func, symbol2)
        b = p1.map(function_call, a)



    # daily summary creation and directory creation
    end_date_new = pd.to_datetime(end_date)
    final_end_date = end_date_new.strftime('%Y-%b-%d')
    final_df = pd.concat(b, ignore_index=True)
    write_to_excel(final_df, 'daily_summary', final_end_date)
    final_df.columns = final_df.columns.str.lower()



    # database insertion logic
    # # database insertion
    # DATABASE_URL = "postgresql+psycopg2://postgres:123456@127.0.0.1:5432/daily_summary_db"
    # engine = create_engine(DATABASE_URL)
    # final_df.to_sql('summary6', engine, index=False, if_exists='append')


    print(datetime.now())