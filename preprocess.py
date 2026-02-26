import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

def preprocess():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, 'dataset', 'KDDTrain+.txt')
    output_x = os.path.join(project_root, 'dataset', 'X.csv')
    output_y = os.path.join(project_root, 'dataset', 'y.csv')

    if not os.path.exists(input_path):
        print(f" Error: Could not find {input_path}")
        return

    print(f" Loading {input_path}...")
    df = pd.read_csv(input_path, names=columns)
    
    
    df['Label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    
    
    le = LabelEncoder()
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(columns=['attack_type', 'difficulty_level', 'Label'])
    y = df['Label']
    
    X.to_csv(output_x, index=False)
    y.to_csv(output_y, index=False)
    print(f" Done! X.csv and y.csv created in 'dataset' folder.")

if __name__ == "__main__":
    preprocess()
