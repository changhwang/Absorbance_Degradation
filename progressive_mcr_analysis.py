import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg
from tqdm import tqdm
from scipy.signal import find_peaks

def normalize_spectra(spectra):
    """스펙트럼 정규화"""
    return spectra / np.max(spectra, axis=1)[:, np.newaxis]

def save_spectral_data(wavelengths, signals, times, result_dir, signal_type='reference'):
    """스펙트럼 데이터를 CSV로 저장"""
    # Raw 데이터
    raw_data = pd.DataFrame(signals, columns=wavelengths)
    raw_data.insert(0, 'Time(h)', times)
    raw_data.to_csv(os.path.join(result_dir, f'{signal_type}_raw_spectra.csv'), index=False)
    
    # Normalized 데이터
    norm_signals = normalize_spectra(signals)
    norm_data = pd.DataFrame(norm_signals, columns=wavelengths)
    norm_data.insert(0, 'Time(h)', times)
    norm_data.to_csv(os.path.join(result_dir, f'{signal_type}_normalized_spectra.csv'), index=False)
    
    # Peak 분석
    peak_times = []
    peak_wavelengths = []
    peak_intensities = []
    norm_peak_intensities = []
    
    for i in range(len(signals)):
        peaks, _ = find_peaks(signals[i], prominence=0.1*np.max(signals[i]))
        if len(peaks) > 0:
            max_peak_idx = peaks[np.argmax(signals[i][peaks])]
        else:
            max_peak_idx = np.argmax(signals[i])
            
        peak_times.append(times[i])
        peak_wavelengths.append(wavelengths[max_peak_idx])
        peak_intensities.append(signals[i][max_peak_idx])
        norm_peak_intensities.append(norm_signals[i][max_peak_idx])
    
    # Peak 데이터 저장
    peak_data = pd.DataFrame({
        'Time(h)': peak_times,
        'Peak_Wavelength(nm)': peak_wavelengths,
        'Peak_Intensity': peak_intensities,
        'Normalized_Peak_Intensity': norm_peak_intensities
    })
    peak_data.to_csv(os.path.join(result_dir, f'{signal_type}_peak_data.csv'), index=False)

def create_result_directory(directory, filename):
    """결과 저장을 위한 디렉토리 생성"""
    base_name = os.path.splitext(filename)[0]
    result_dir = os.path.join(directory, base_name)
    try:
        os.makedirs(result_dir, exist_ok=True)
        print(f"\nCreated directory: {result_dir}")
    except Exception as e:
        print(f"\nError creating directory: {str(e)}")
        result_dir = os.path.join(directory, 'results')
        os.makedirs(result_dir, exist_ok=True)
        print(f"Using alternative directory: {result_dir}")
    return result_dir

def progressive_mcr_analysis(df, directory, filename):
    """시간에 따른 점진적 MCR-ALS 분석"""
    try:
        result_dir = create_result_directory(directory, filename)
        wavelengths = df["Wavelength"].values
        time_columns = df.columns[1:]
        times = np.array([float(col) for col in time_columns])
        
        print(f"Processing {filename}...")
        print(f"Data shape: {df.shape}")
        print(f"Time points: {len(times)}")
        print(f"Column names: {df.columns.tolist()}")
        
        # 데이터 준비
        spectra_matrix = df.iloc[:, 1:].values.T
        
        # MCR-ALS 분석
        window_size = 4  # 기본 윈도우 크기
        reference_signals = []
        degraded_signals = []
        concentrations = []
        successful_analyses = 0
        
        for i in tqdm(range(window_size, len(times))):
            try:
                # 현재 윈도우의 데이터만 사용
                current_window = spectra_matrix[i-window_size:i+1]
                
                # 현재 윈도우에 대한 초기 추측값 계산
                current_initial = np.vstack([
                    current_window[0],  # 현재 윈도우의 첫 번째 스펙트럼
                    current_window[-1]  # 현재 윈도우의 마지막 스펙트럼
                ])
                
                # 새로운 MCR 인스턴스로 독립적인 분석
                mcr_new = McrAR(
                    c_regr='OLS',
                    st_regr='NNLS',
                    c_constraints=[ConstraintNonneg()],
                    max_iter=1000,
                    tol_increase=1.2,
                    tol_n_increase=30,
                    tol_err_change=1e-8
                )
                
                mcr_new.fit(current_window, ST=current_initial)
                reference_signals.append(mcr_new.ST_[0])
                degraded_signals.append(mcr_new.ST_[1])
                concentrations.append(mcr_new.C_)
                successful_analyses += 1
                
            except Exception as e:
                print(f"\nError in analysis {i}: {str(e)}")
                continue
        
        print(f"\nSuccessful analyses: {successful_analyses} out of {len(times)-window_size}")
        
        # 결과를 numpy 배열로 변환
        reference_signals = np.array(reference_signals)
        degraded_signals = np.array(degraded_signals)
        
        print("\nArray shapes after MCR-ALS:")
        print(f"reference_signals: {reference_signals.shape}")
        print(f"degraded_signals: {degraded_signals.shape}")
        print(f"times: {times[window_size:successful_analyses+window_size].shape}")
        
        # 데이터 저장
        save_spectral_data(wavelengths, reference_signals, 
                          times[window_size:successful_analyses+window_size], 
                          result_dir, 'reference')
        save_spectral_data(wavelengths, degraded_signals, 
                          times[window_size:successful_analyses+window_size], 
                          result_dir, 'degraded')
        
        # Concentration profiles 저장
        try:
            last_concentration = concentrations[-1]
            times_for_conc = times[window_size:window_size+len(last_concentration)]
            
            concentration_data = pd.DataFrame({
                'Time(h)': times_for_conc,
                'Reference_Concentration': last_concentration[:, 0],
                'Degraded_Concentration': last_concentration[:, 1]
            })
            concentration_data.to_csv(os.path.join(result_dir, 'concentration_profiles.csv'), index=False)
            print("\nConcentration profiles saved successfully")
        except Exception as e:
            print(f"\nError saving concentration profiles: {str(e)}")
            print(f"Times shape: {times_for_conc.shape}, Concentration shape: {last_concentration.shape}")
        
        # Reference vs Degraded 비교
        print("\nReference vs Degraded comparison:")
        print(f"Reference first timepoint max: {np.max(reference_signals[0])}")
        print(f"Degraded first timepoint max: {np.max(degraded_signals[0])}")
        print(f"Normalized reference first timepoint max: {np.max(normalize_spectra(reference_signals)[0])}")
        print(f"Normalized degraded first timepoint max: {np.max(normalize_spectra(degraded_signals)[0])}")
        
        # Overlay Plots
        plt.figure(figsize=(15, 10))
        
        # Reference Raw
        ax1 = plt.subplot(2, 2, 1)
        for i, spectrum in enumerate(reference_signals):
            plt.plot(wavelengths, spectrum, color=plt.cm.viridis(i/len(reference_signals)))
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(min(times), max(times)), cmap='viridis'), 
                    ax=ax1, label='Time (h)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Raw Reference Signals')
        
        # Reference Normalized
        ax2 = plt.subplot(2, 2, 2)
        normalized_reference = normalize_spectra(reference_signals)
        for i, spectrum in enumerate(normalized_reference):
            plt.plot(wavelengths, spectrum, color=plt.cm.viridis(i/len(normalized_reference)))
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(min(times), max(times)), cmap='viridis'), 
                    ax=ax2, label='Time (h)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.title('Normalized Reference Signals')
        
        # Degraded Raw
        ax3 = plt.subplot(2, 2, 3)
        for i, spectrum in enumerate(degraded_signals):
            plt.plot(wavelengths, spectrum, color=plt.cm.viridis(i/len(degraded_signals)))
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(min(times), max(times)), cmap='viridis'), 
                    ax=ax3, label='Time (h)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Raw Degraded Signals')
        
        # Degraded Normalized
        ax4 = plt.subplot(2, 2, 4)
        normalized_degraded = normalize_spectra(degraded_signals)
        for i, spectrum in enumerate(normalized_degraded):
            plt.plot(wavelengths, spectrum, color=plt.cm.viridis(i/len(normalized_degraded)))
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(min(times), max(times)), cmap='viridis'), 
                    ax=ax4, label='Time (h)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.title('Normalized Degraded Signals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'overlay_plots.png'))
        plt.close()
        
        # 3D Surface Plots
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.4, hspace=0.3)
        
        # Reference Signal Surface
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        wavelength_mesh, time_mesh = np.meshgrid(wavelengths, times[window_size:successful_analyses+window_size])
        surf1 = ax1.plot_surface(wavelength_mesh, time_mesh, reference_signals, cmap='viridis')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Time (h)')
        ax1.set_zlabel('Intensity')
        ax1.set_title('Raw Reference Signal Evolution')
        plt.colorbar(surf1, ax=ax1, label='Intensity')
        
        # Normalized Reference Signal Surface
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        surf2 = ax2.plot_surface(wavelength_mesh, time_mesh, normalized_reference, cmap='viridis')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Time (h)')
        ax2.set_zlabel('Normalized Intensity')
        ax2.set_title('Normalized Reference Signal Evolution')
        plt.colorbar(surf2, ax=ax2, label='Normalized Intensity')
        
        # Degraded Signal Surface
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        surf3 = ax3.plot_surface(wavelength_mesh, time_mesh, degraded_signals, cmap='viridis')
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Time (h)')
        ax3.set_zlabel('Intensity')
        ax3.set_title('Raw Degraded Signal Evolution')
        plt.colorbar(surf3, ax=ax3, label='Intensity')
        
        # Normalized Degraded Signal Surface
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        surf4 = ax4.plot_surface(wavelength_mesh, time_mesh, normalized_degraded, cmap='viridis')
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Time (h)')
        ax4.set_zlabel('Normalized Intensity')
        ax4.set_title('Normalized Degraded Signal Evolution')
        plt.colorbar(surf4, ax=ax4, label='Normalized Intensity')
        
        plt.savefig(os.path.join(result_dir, 'surface_plots.png'), bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        raise

def process_directory(directory):
    """디렉토리 내의 모든 CSV 파�� 처리"""
    print(f"Processing files in: {directory}\n")
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and not filename.startswith('kinetics'):
            print(f"\nProcessing file: {filename}")
            try:
                # 먼저 헤더를 읽어서 시간 포인트 확인
                with open(os.path.join(directory, filename), 'r') as f:
                    header = [next(f) for _ in range(11)]
                    time_points = header[10].strip().split(',')[2:]
                
                # CSV 파일 읽기
                df = pd.read_csv(
                    os.path.join(directory, filename), 
                    skiprows=11,
                    names=['Index', 'Wavelength'] + time_points
                )
                
                # 'Index' 열만 제거
                df = df.drop('Index', axis=1)
                
                # 데이터 확인
                print("\nProcessed DataFrame:")
                print(df.head())
                print("\nColumns:", df.columns.tolist())
                
                progressive_mcr_analysis(df, directory, filename)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                import traceback
                print(traceback.format_exc())
    print("\nAnalysis complete!")

if __name__ == "__main__":
    data_directory = os.path.join("data", "uvvis", "PDCBT_1day_data_success")
    process_directory(data_directory)