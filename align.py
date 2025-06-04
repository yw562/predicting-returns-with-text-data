import pandas as pd

def align_stock_data(df):
    # 转换日期格式，错误转换为 NaT
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

    # 过滤掉任何关键字段缺失的行
    required_cols = ['TWEET', 'STOCK', 'DATE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
    df = df.dropna(subset=required_cols)

    df = df.reset_index(drop=True)
    print("对齐完成，有效样本数:", len(df))
    print("对齐后前五行数据：")
    print(df.head())
    return df


if __name__ == "__main__":
    # 示例用法：读取文件，执行对齐
    file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"
    df = pd.read_csv(file_path)
    print(f"原始样本数: {len(df)}")
    
    df_aligned = align_stock_data(df)


# import pandas as pd

# def align_stock_data(df):
#     # 转换日期格式，错误转换为 NaT
#     df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

#     # 过滤掉任何关键字段缺失的行
#     required_cols = ['TWEET', 'STOCK', 'DATE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
#     df = df.dropna(subset=required_cols)

#     df = df.reset_index(drop=True)
#     print("对齐完成，有效样本数:", len(df))
#     return df


# if __name__ == "__main__":
#     # 示例用法：读取文件，执行对齐
#     file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"
#     df = pd.read_csv(file_path)
#     print(f"原始样本数: {len(df)}")
    
#     df_aligned = align_stock_data(df)
