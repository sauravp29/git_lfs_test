import copy


def process_score_df(score_df, header_df):
    print('processing score data')


    score_df = score_df.merge(header_df,on='enquiryControlNumber',
                              how='inner',suffixes=('', '_y'))
    

    if 'Score_Date' in score_df.columns:
        score_df = score_df.sort_values('Score_Date',kind='mergesort').drop_duplicates(keep='last')
        # score_df = score_df.drop(columns=['Score_Date'])
        
    elif 'Score' in score_df.columns:
        score_df = score_df.sort_values('CIBIL_Report_Date',kind='mergesort').drop_duplicates(
            'enquiryControlNumber', keep='last')
      
    return score_df
