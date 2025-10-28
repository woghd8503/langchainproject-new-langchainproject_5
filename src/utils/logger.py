# ---------------------- 로그 기록 모듈 ---------------------- #

import sys                                       # 시스템 관련 기능 모듈
from datetime import datetime                    # 현재 시간 가져오기 모듈
from tqdm import tqdm                            # 진행률 표시 모듈

# ---------------------- Logger 클래스 정의 ---------------------- #
class Logger:                                    # Logger 클래스 정의
    """
    로그를 파일에 저장하고, 표준 출력(stdout)과 표준 에러(stderr)를
    로그 파일로 리디렉션하는 기능이 추가된 Logger 클래스
    """
    # 초기화 함수 정의
    def __init__(self, log_path: str, print_also: bool = True):
        self.log_path = log_path                 # 로그 파일 경로 저장
        self.print_also = print_also             # 콘솔 출력 여부 저장
        # 원본 표준 출력을 저장해 둡니다.
        self.original_stdout = sys.stdout        # 원본 표준 출력 저장
        self.original_stderr = sys.stderr        # 원본 표준 에러 저장
        # 로그 파일을 열고, 라인 버퍼링을 사용합니다.
        self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)  # 로그 파일 열기

    
    # 로그 기록 함수 정의
    def write(self, message: str, print_also: bool = True, print_error: bool = False):
        """
        로그 메시지를 파일에 기록하고,
        print_also=True일 경우 콘솔에도 출력합니다.
        """
        # 메시지 앞뒤 공백을 제거하고, 개행 문자가 없으면 추가합니다.
        message = message.strip()                # 메시지 공백 제거
        if not message:                          # 메시지가 비어있으면
            return                               # 함수 종료
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 현재 시간 타임스탬프 생성
        line = f"{timestamp} | {message}\n"      # 타임스탬프와 메시지 결합
        
        self.log_file.write(line)                # 로그 파일에 기록
        
        # 콘솔 출력 옵션이 활성화된 경우
        if self.print_also and print_also:
            # 에러 메시지인 경우
            if print_error:
                # 재귀 호출을 피하기 위해 원본 표준 출력을 사용합니다.
                self.original_stdout.write(f"\033[91m{line}\033[0m")  # 빨간색으로 에러 출력
            # 일반 메시지인 경우
            else:
                self.original_stdout.write(line) # 일반 출력
    
    
    # 플러시 함수 정의
    def flush(self):
        """
        스트림 인터페이스에 필요한 flush 메서드입니다.
        """
        self.log_file.flush()   # 로그 파일 버퍼 플러시


    # 리다이렉션 시작 함수 정의
    def start_redirect(self):
        """
        표준 출력(stdout)과 표준 에러(stderr)를 이 로거 인스턴스로 리다이렉션합니다.
        """
        self.write(">> 표준 출력 및 오류를 로그 파일로 리디렉션 시작", print_also=True)  # 리다이렉션 시작 로그
        sys.stdout = self   # 표준 출력을 로거로 리디렉션
        sys.stderr = self   # 표준 에러를 로거로 리디렉션


    # 리다이렉션 중지 함수 정의
    def stop_redirect(self):
        """
        표준 출력(stdout)과 표준 에러(stderr)를 원상 복구합니다.
        """
        self.write(">> 로그 리디렉션 중료.", print_also=True)  # 리다이렉션 중지 로그
        sys.stdout = self.original_stdout        # 표준 출력 원상 복구
        sys.stderr = self.original_stderr        # 표준 에러 원상 복구
    
    
    # tqdm 리다이렉션 함수 정의
    def tqdm_redirect(self):
        """tqdm.write를 이 로거 인스턴스로 리다이렉션합니다."""
        # tqdm 호환 래퍼 함수 정의
        def tqdm_write_wrapper(s, file=None, end="\n", nolock=False):
            self.write(s)                        # 메시지를 로거로 전달
        tqdm.write = tqdm_write_wrapper          # tqdm 출력을 래퍼 함수로 리다이렉션

    
    # 로거 종료 함수 정의
    def close(self):
        """
        로그 파일을 닫습니다.
        """
        self.log_file.close() # 로그 파일 닫기