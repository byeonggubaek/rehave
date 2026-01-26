#############################################################################################################
# 작성일 : 2026-01-16  
# 작성자 : 백병구  
# 내용 : 웹켐으로 환자의 상지 움직임을 분석하여 환자 상태에 맞는 운동방법을 제안해 준다.   
#############################################################################################################
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import streamlit as st
from streamlit.components.v1 import html
from cap_from_youtube import cap_from_youtube
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import random
import time
#############################################################################################################
# Patient 객체:  환자
#############################################################################################################
patients= []
class Patient:
    def __init__(self, id, name, age, sex):
        self.id = id  # 아이디 
        self.name = name # 이름
        self.age = age    # 나이 
        self.sex = sex    # 성별 
    def get_cate(self): 
        if self.age >= 60:
            return self.sex + "60"
        elif self.age >= 40:
            return self.sex + "40"
        elif self.age >= 20:
            return self.sex + "20"
        else:
            return self.sex + "10"

#############################################################################################################
# GradeChecker 객체: 등급 검사기 나이대, 성별에 따라 외전 각도의 기준이 다를 수 있기 때문에  
#############################################################################################################
standards = []
class GradeChecker:
    def __init__(self, age, sex, std_angle):
        self.age = age  # 나이대
        self.sex = sex    # 성별
        self.std_angle = std_angle # 기준 외전 

    def check_grade(self, angel):
        if angel >= 160:    
            return 'G' # 양호
        elif angel >= 120 and angel < 160:
            return 'I' # 개선
        else:
            return "C" # 주의
        
#############################################################################################################
# RecommandWorkout 객체: 추천운동  
#############################################################################################################
rec_workouts = []
class RecommandWorkout:
    def __init__(self, grade, workout):
        self.grade = grade  # 추천운동
        self.workout = workout  # 추천운동

    @staticmethod
    def recommand(grade):
        rec_workout = [obj for obj in rec_workouts if obj.grade == grade]    
        if rec_workout is not None and len(rec_workout) >= 1:
            return random.choice(rec_workout).workout
        else:
            return ""
        
#############################################################################################################
# 공용 함수 
#############################################################################################################
# grade코드를 사람이 인식할 수 있는 문자열로 리턴한다. 
def get_grade(grade):
    match grade:
        case "G":
            return "양호"
        case "I":
            return "개선"
        case _:
            return "주의"
# 두점을 지나는 직선과 수직선의 각도를 구한다. 
def get_angel(p1, p2):
    vec = np.array(p2) - np.array(p1)
    vertical = np.array([0, 1])
    cosine = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
    angle = abs(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    if angle > 180:
        angle = 180
    return angle
# 환자의 범주와 외전의 각도를 이용해 환자의 운동 등급을 리턴한다. 
def check(cate, angel):
    sex = cate[0]
    age = int(cate[1:])
    checker = [obj for obj in standards if obj.age == age and obj.sex == sex]
    if checker:  # 리스트가 비어있지 않으면
        return checker[0].check_grade(angel)  # 첫 번째 객체 사용
    else:
        return "X"
def play(url):
    cap = cap_from_youtube(url)
    cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)  # 크기 조절 허용
    cv2.moveWindow('YOLO Detection', 100, 100)  # x=100, y=100 위치

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
#############################################################################################################
# 각단계의 처리 
#############################################################################################################
# 초기화 
def init_system():
    # bootstrap 적용 및 세션 상태 초기화
    st.markdown("""`
    <!DOCTYPE html>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)
    # CSS 적용
    with open("main.css", "r", encoding='utf-8') as f:
        css = f.read()    
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    # print(f"시스템 초기화 전: {st.session_state}")
    # 세션 상태 초기화
    if 'user_id' not in st.session_state or st.session_state.user_id == "":
        st.session_state.user_id = ""        
        st.session_state.log_in = False
        st.session_state.current_patient = None
        st.session_state.running = False
        st.session_state.result = None        
    # print(f"시스템 초기화 후: {st.session_state}")
               
    # 환자 및 등급 검사기 초기화        
    patients.append(Patient('001', '문정인', 18, 'F'))
    patients.append(Patient('002', '조성현', 25, 'M'))
    patients.append(Patient('003', '이호재', 25, 'M'))
    patients.append(Patient('004', '백병구', 25, 'M'))

    standards.append(GradeChecker(10, 'M', 180))
    standards.append(GradeChecker(10, 'F', 180))
    standards.append(GradeChecker(20, 'M', 180))
    standards.append(GradeChecker(20, 'F', 180))
    standards.append(GradeChecker(40, 'M', 176))
    standards.append(GradeChecker(40, 'F', 177))
    standards.append(GradeChecker(60, 'M', 170))
    standards.append(GradeChecker(60, 'F', 177))

    rec_workouts.append(RecommandWorkout('C', "어깨 재활 운동"))
    rec_workouts.append(RecommandWorkout('C', "어깨 으쓱 돌리기 운동"))
    rec_workouts.append(RecommandWorkout('C', "말린어깨 라운드숄더 쫙펴지는 운동"))    
    rec_workouts.append(RecommandWorkout('I', "벽 짚고 팔 올리기 운동"))
    rec_workouts.append(RecommandWorkout('G', "Y-Raise"))
       
# login Callback 함수
def login_callback(user_id):
    patient = [obj for obj in patients if obj.id == user_id or obj.name == user_id]
    if patient is not None and len(patient) >= 1:
        st.session_state.user_id = user_id
        st.session_state.log_in = True
        st.session_state.current_patient = patient[0]
        st.session_state.running = False
        st.session_state.result = None
        st.rerun()
    else:
        st.error(f'안녕하세요! {user_id}님. 아이디를 찾을 수 없습니다. 다시 입력하세요.')
# logoff Callback 함수
def logoff_callback():
    if st.session_state.running == True:
        return
    st.session_state.user_id = ""
    st.rerun()
# reset Callback 함수
def reset_callback():
    if st.session_state.running == True:
        return
    st.session_state.running = False 
    st.session_state.result = None          
    st.rerun()    
# 외전분석 Callback 함수    
def anal_callback(pat):
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    window_name = 'YOLO Pose Angles'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) 

    max_l_shoulder_angle = 0
    max_r_shoulder_angle = 0
    start = time.time()    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (800, 600))
        results = model(frame, conf=0.2, verbose=False)
        annotated = results[0].plot() if results else frame
        
        # 텍스트 배경 (검은색 사각형)
        x_start = 10 
        x_end = 250
        text_height = 30
        y_start = 30
        y_end = 100
        cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)  # 왼쪽 상단 영역 

        for r in results:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            confs = r.keypoints.conf[0].cpu().numpy()

            if len(keypoints) > 0:
                cv2.putText(annotated, f'Left:', (x_start + 10, y_start + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)            
                # 5: left_shoulder
                # 7: left_elbow
                l_points = [5,7] # 왼손  
                if np.all(confs[l_points] > 0.2):
                    l_shoulder, l_elbow = keypoints[5], keypoints[7]
                    l_shoulder_angle = get_angel(l_shoulder, l_elbow)
                    if l_shoulder_angle > max_l_shoulder_angle :
                        max_l_shoulder_angle = l_shoulder_angle
                    # 깨짐 없는 텍스트 (숫자만 + 크기 조정)
                    cv2.putText(annotated, f'{int(l_shoulder_angle)} {int(max_l_shoulder_angle)}', (x_start + 10 + 80, y_start + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    # 수직선
                    cv2.line(annotated, (int(l_shoulder[0]), int(l_shoulder[1])), 
                            (int(l_shoulder[0]), 600), (0,0,255), 1)
                    
                cv2.putText(annotated, f'Right:', (x_start + 10, y_start + 30 + text_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)                
                # 6: right_shoulder
                # 8: right_elbow
                r_points = [6,8] # 오른손                 
                if np.all(confs[r_points] > 0.2):
                    r_shoulder, r_elbow = keypoints[6], keypoints[8]
                    r_shoulder_angle = get_angel(r_shoulder, r_elbow)
                    if r_shoulder_angle > max_r_shoulder_angle :
                        max_r_shoulder_angle = r_shoulder_angle                    
                    # 깨짐 없는 텍스트 (숫자만 + 크기 조정)
                    cv2.putText(annotated, f'{int(r_shoulder_angle)} {int(max_r_shoulder_angle)}', (x_start + 10 + 80, y_start + 30 + text_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    # 수직선
                    cv2.line(annotated, (int(r_shoulder[0]), int(r_shoulder[1])), 
                            (int(r_shoulder[0]), 700), (0,0,255), 1)       
        
        end = time.time()
        elapsed = end - start             
        if 15 - int(elapsed) > 9:
            cv2.putText(annotated, f'{15 - int(elapsed)}', (x_start + 10 + 630, y_start + 180), 
                    cv2.FONT_HERSHEY_PLAIN, 7, (242,108,5), 3)        
        else:
            cv2.putText(annotated, f'{15 - int(elapsed)}', (x_start + 10 + 700, y_start + 180), 
                    cv2.FONT_HERSHEY_PLAIN, 7, (242,108,5), 3)        
        cv2.imshow(window_name, annotated)                    
        if elapsed > 15:  # 30초 동안 측정q
            break 
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()

    cv2.destroyAllWindows()
    # 바꾸었어요
    
    # 기존 running 함수 (환자 분석 화면)
    grade = check(pat.get_cate(), min(max_l_shoulder_angle, max_r_shoulder_angle))  # check, get_grade 함수 정의 필요

    return (grade, min(max_l_shoulder_angle, max_r_shoulder_angle))
# 유튜브 Callback 함수
def youtube_callback():
    grade, max_angle = st.session_state.result      
    options = Options()
    options.add_experimental_option("detach", True)  # 브라우저 유지    
    driver = webdriver.Chrome(options=options)
    workout = RecommandWorkout.recommand(grade)
    if workout != "":
        driver.get("https://www.youtube.com/results?search_query=" + workout)
    
# 판정 
def judge_print(patient):
    grade, max_angle = st.session_state.result  
    st.write("")
    match grade:
        case "G": # 양호
            st.success(f"{patient.name} 회복이 필요한 어깨 벌림 값은 {int(max_angle)} 이고 등급은 {get_grade(grade)} 입니다. 리듬에 맞추어 어깨를 움직여 보세요.")
            st.video("g.mp4", autoplay=True)
        case "I": # 개선   
            st.info(f"{patient.name} 회복이 필요한 어깨 벌림 값은 {int(max_angle)} 이고 등급은 {get_grade(grade)} 입니다. 아래 운동을 따라해 보세요.")
            st.video("i.mp4", autoplay=True)
        case "C": # 주의   
            st.error(f"{patient.name} 회복이 필요한 어깨 벌림 값은 {int(max_angle)} 이고 등급은 {get_grade(grade)} 입니다. 아래 운동을 따라해 보세요.")
            st.video("c.mp4", autoplay=True)
        case _:
            pass
    st.session_state.running = False      
# 기준 
def standard_print():    
    st.write("")
    st.markdown("""
    <div class="stand_table mb-4 pt-2 border-top border-color-1">
        <table style="width:100%">
        <tr>
            <th>가동범위(굴곡기준)</th>
            <th>판정</th>
            <th>추천 솔루션(운동)</th>
            <th>운동 목적</th>
        </tr>
        <tr>
            <td>  120° 미만</td>
            <td id="Caution"> 주의 </td>
            <td> 어깨 으쓱 돌리기 운동</td>
            <td> 중력 부담 최소화, 기초 가동성 확보</td>
        </tr>
        <tr>
            <td>  120° ~ 160°</td>
            <td id="Improve"> 개선 </td>
            <td> 벽 짚고 팔 올리기 (Wall Slides)</td>
            <td> 도구를 활용한 가동 범위 확장</td>
        </tr>
        <tr>
            <td>  160° 이상</td>
            <td id="Good"> 양호 </td>
            <td> Y-Raise (맨몸 근력 운동)</td>
            <td> 가동범위 유지 및 주변 근육 강화</td>
        </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)  
    st.session_state.standard = False
# 로그인 화면
def login():
    st.markdown("""
    <div class="title mb-4 border-bottom border-color-1">
        RE : HAVE
    </div>
    """, unsafe_allow_html=True)    
    st.image("system.png")        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card" style="width: 18rem;">
            <div class="card-body">
                <h5 class="card-title color-2">대칭 기반 재활 시스템</h5>
                <p class="card-text">웹캠으로 환자의 어깨 외전 각도를 YOLO11n-pose로 실시간 분석하여, 개인별 최적의 운동 프로그램을 자동 제안합니다</p>
            </div>
            </div>
        </div>
        """, unsafe_allow_html=True)  
    with col2:
        user_id = st.text_input("아이디 입력:", value=st.session_state.user_id, width=300)
        if st.button("로그인"):
            login_callback(user_id)
# 분석 화면,
def action():
    patient = st.session_state.current_patient        
    st.markdown("""
    <div class="title mb-4 border-bottom border-color-1">
        RE : HAVE
    </div>
    """, unsafe_allow_html=True)            
    if (st.session_state.current_patient is not None):
        result = None
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card" style="width: 18rem;">
                <div class="card-body">
                    <h5 class="card-title color-2">외전 분석</h5>
                    <p class="card-text">안녕하세요! <span class="card-text color-2">""" + patient.name + """님.</span> AI 치료사가 회원님의 증상을 분석해서 알맞은 운동 방법을 소개해 드립니다.</p>
                </div>
                </div>
            </div>
            """, unsafe_allow_html=True)  
        with col2:
            col3, col4, col5 = st.columns(3)
            with col3:
                # 분석시작 버튼
                if st.button("분 석", disabled=st.session_state.running):
                    st.session_state.running = True      
                    st.rerun()  # 즉시 재실행으로 버튼 비활성화                                          
                if st.session_state.running:
                    with st.spinner("분석 중..."):
                        result = anal_callback(st.session_state.current_patient)  
                        st.session_state.result = result
                        st.session_state.running = False
                        st.rerun()
                if st.session_state.result is not None:
                    # 유튜브 버튼                    
                    if st.button("추 천", disabled=st.session_state.running):
                        youtube_callback()                                      
            with col4:        
                # 재시작 버튼                    
                if st.button("초기화", disabled=st.session_state.running):
                    reset_callback()       
            with col5:        
                # 로그아웃 버튼                    
                if st.button("로그아웃", disabled=st.session_state.running):
                    logoff_callback()
        if st.session_state.result is not None:
            judge_print(patient)
        else:     
            standard_print()      
# system 가동
def run_system():
    # 로그인 화면 
    if not st.session_state.log_in:
        login()
    # 외전분석 화면                 
    else:
        action()

#############################################################################################################
# main 프로그램 실행 
#############################################################################################################
def main():
    # 초기화 
    init_system()    
    # system 가동
    run_system()  

if __name__ == "__main__":
    main()