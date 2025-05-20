import streamlit as st
import pandas as pd
import numpy as np
import cv2
from datetime import date
import uuid

from recognizer import (
    load_database, save_database,
    preprocess_face, get_embedding,
    recognize_face, detect_faces_from_frame
)
from utils import (
    ensure_csv_files, add_class, load_classes,
    log_attendance, view_attendance_log,
    get_registered_students, delete_student_by_name, delete_student_by_uuid
)

st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("🧑‍💼 Face Recognition Attendance System")

ensure_csv_files()
database = load_database()

menu = st.sidebar.radio("📋 Menu", ["Home", "Take Attendance", "Add Student", "Add Class", "View Students", "View Logs"])

# Application modes
if menu == "Home":
    st.header("Face Recognition Attendance System")
    st.write("""
    Welcome to the Face Recognition Attendance System! This application allows you to:
    
    1. Register new people for face recognition
    2. Take attendance using your webcam
    3. View attendance records
    4. Manage registered people
    
    Select a mode from the sidebar to get started.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Register New Person**: Add people to the face recognition database")
    with col2:
        st.info("📸 **Take Attendance**: Use your webcam to mark attendance")
    
    col3, col4 = st.columns(2)
    with col3:
        st.info("📊 **View Attendance**: Check attendance records and reports")
    with col4:
        st.info("👥 **View Registered People**: Manage people in the database")

# --- 📸 Take Attendance ---
elif menu == "Take Attendance":
    st.header("📸 Take Attendance")
    classes_df = load_classes()

    if len(classes_df) == 0:
        st.warning("⚠️ No class found. Please add a class first.")
    else:
        class_names = classes_df["Class Name"] + " (" + classes_df["Date"] + ")"
        selected = st.selectbox("Choose Class", class_names)
        selected_class = selected.split(" (")[0]

        # Initialize session state for recognized faces
        if 'recognized_faces' not in st.session_state:
            st.session_state.recognized_faces = []
            st.session_state.current_frame = None

        uploaded_img = st.camera_input("Take a photo for attendance")

        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            faces, rgb_frame = detect_faces_from_frame(frame)
            
            # Store the frame for later use
            st.session_state.current_frame = frame

            if len(faces) == 0:
                st.error("❌ No face detected. Try again.")
            else:
                # Clear previous recognitions
                st.session_state.recognized_faces = []
                
                # Process all detected faces
                for face in faces:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    x2, y2 = x + w, y + h
                    face_img = rgb_frame[y:y2, x:x2]
                    processed_face = preprocess_face(face_img)
                    
                    if processed_face is None:
                        continue
                        
                    embedding = get_embedding(processed_face)
                    name, confidence = recognize_face(embedding, database)
                    
                    # Store recognized face information
                    st.session_state.recognized_faces.append({
                        'name': name,
                        'confidence': confidence,
                        'box': face['box']
                    })
                
                # Display recognized faces
                if st.session_state.recognized_faces:
                    st.subheader("Recognized Students:")
                    
                    # Create a DataFrame to display recognized faces
                    recognized_data = []
                    for i, face_data in enumerate(st.session_state.recognized_faces):
                        status = "✅ Known" if face_data['name'] != "Unknown" else "❓ Unknown"
                        confidence_score = f"{(1 - face_data['confidence']) * 100:.1f}%" if face_data['name'] != "Unknown" else "N/A"
                        
                        recognized_data.append({
                            "Student": face_data['name'],
                            "Status": status,
                            "Confidence": confidence_score
                        })
                    
                    # Display as a table
                    recognized_df = pd.DataFrame(recognized_data)
                    st.table(recognized_df)
                    
                    if st.button("✅ Mark Attendance", key="mark_attendance_button"):
                        attendance_count = 0
                        for face_data in st.session_state.recognized_faces:
                            if face_data['name'] != "Unknown":
                                # Log attendance for known faces only
                                log_attendance(face_data['name'], selected_class)
                                attendance_count += 1
                        
                        if attendance_count > 0:
                            st.success(f"Attendance recorded successfully for {attendance_count} student(s)!")
                        else:
                            st.warning("No known students to mark attendance for.")
                        
                        # Explicitly reset session state for next capture
                        st.session_state.recognized_faces = []
                        st.session_state.current_frame = None
                        
                        # Clear any image placeholder if used
                        if 'image_placeholder' in st.session_state:
                            st.session_state.image_placeholder = None
                        
                        # Rerun to refresh the app
                        st.rerun()

# --- ➕ Add Student ---
elif menu == "Add Student":
    st.header("➕ Add New Student")
    col1, col2 = st.columns(2)
    name = col1.text_input("Student Name")
    nim = col2.text_input("Student ID (NIM)")
    
    # Store captured images and embeddings in session state
    if 'captured_embeddings' not in st.session_state:
        st.session_state.captured_embeddings = []
        st.session_state.image_count = 0
        st.session_state.save_complete = False
        st.session_state.current_image = None
    
    # Show progress
    if st.session_state.image_count > 0:
        st.info(f"📸 {st.session_state.image_count}/5 images captured")
        # Ensure progress value is between 0.0 and 1.0
        progress_value = min(st.session_state.image_count / 5, 1.0)
        progress = st.progress(progress_value)
    
    # Camera input (only show when not complete and fewer than 5 images)
    if not st.session_state.save_complete and st.session_state.image_count < 5:
        uploaded_img = st.camera_input("Take a clear photo (You'll need 5 different angles)")
    
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            # Store current image in session state instead of processing immediately
            st.session_state.current_image = {
                'frame': frame
            }
            
            # Add a confirm button without showing preview
            if st.button("Confirm Face Capture"):
                faces, rgb_frame = detect_faces_from_frame(frame)
                
                if len(faces) == 0:
                    st.error("❌ No face detected. Try again.")
                elif len(faces) > 1:
                    st.error("❌ Multiple faces detected. Please ensure only one person is in frame.")
                else:
                    face = faces[0]
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    x2, y2 = x + w, y + h
                    face_img = rgb_frame[y:y2, x:x2]
                    processed_face = preprocess_face(face_img)
                    
                    if processed_face is None:
                        st.error("❌ Face preprocessing failed. Try again with better lighting.")
                    else:
                        embedding = get_embedding(processed_face)
                        st.session_state.captured_embeddings.append(embedding)
                        st.session_state.image_count = min(st.session_state.image_count + 1, 5)
                        
                        # Clear current image
                        st.session_state.current_image = None
                        
                        # Set a flag to indicate we need to reset camera
                        if 'reset_camera' not in st.session_state:
                            st.session_state.reset_camera = True
                        else:
                            st.session_state.reset_camera = True
                        
                        # Display success for this capture
                        st.success(f"✅ Image {st.session_state.image_count} captured successfully!")
                        
                        # Reset the entire page to clear the camera
                        st.rerun()
    
    # Save button - only enabled when we have 5 images
    save_disabled = st.session_state.image_count < 5 or not name
    
    if st.button("Save Student", disabled=save_disabled):
        if not name:
            st.warning("Please enter a student name.")
        else:
            # Average the embeddings for better recognition
            avg_embedding = np.mean(st.session_state.captured_embeddings, axis=0)
            
            # Generate a unique ID for this student
            student_uuid = str(uuid.uuid4())
            
            # Save to database
            database["names"].append(name)
            database["embeddings"].append(avg_embedding)
            
            # Initialize or update uuids list
            if "uuids" not in database:
                database["uuids"] = []
            database["uuids"].append(student_uuid)
            
            # Initialize or update nims list
            if "nims" not in database:
                database["nims"] = ["" for _ in range(len(database["names"])-1)]
            database["nims"].append(nim)
            
            save_database(database)
            
            # Mark complete and show success
            st.session_state.save_complete = True
            success_msg = f"✅ {name}"
            if nim:
                success_msg += f" (NIM: {nim})"
            success_msg += " successfully added with 5 different face captures!"
            st.success(success_msg)
            
    # Reset button - visible after saving or when images are captured
    if st.session_state.save_complete or st.session_state.image_count > 0:
        if st.button("Register Another Student"):
            # Reset all session state
            st.session_state.captured_embeddings = []
            st.session_state.image_count = 0
            st.session_state.save_complete = False
            st.session_state.current_image = None
            st.rerun()

# --- 📚 Add Class ---
elif menu == "Add Class":
    st.header("📚 Add Class")
    class_name = st.text_input("Class Name")
    class_date = st.date_input("Class Date", value=date.today())

    if st.button("Add"):
        if class_name:
            add_class(class_name, str(class_date))
            st.success("✅ Class added!")
        else:
            st.warning("Class name is required.")

# --- 👥 View Students ---
elif menu == "View Students":
    st.header("👥 Registered Students")
    
    if len(database["names"]) > 0:
        # Create a dataframe to display student information
        student_data = []
        for i, name in enumerate(database["names"]):
            nim = database["nims"][i] if "nims" in database and i < len(database["nims"]) else ""
            student_data.append({
                "Name": name,
                "NIM": nim,
                "UUID": database["uuids"][i]
            })
            
        # Create a container to allow refreshing this section
        student_container = st.container()
        
        with student_container:
            # Display student information in columns
            for i, student in enumerate(student_data):
                col1, col2, col3 = st.columns([2, 2, 1])
                col1.write(student["Name"])
                col2.write(student["NIM"])
                
                if col3.button("❌ Delete", key=f"delete_{student['UUID']}"):
                    # Delete student from database using UUID
                    success, student_name = delete_student_by_uuid(database, student["UUID"])
                    if success:
                        save_database(database)
                        st.success(f"Student {student_name} deleted successfully!")
                        # Rerun the app to refresh the list
                        st.rerun()
                    else:
                        st.error("Failed to delete student. Please try again.")
    else:
        st.info("No students registered.")

# --- 📝 View Logs ---
elif menu == "View Logs":
    st.header("📝 Attendance Logs")
    
    # Load attendance logs and classes
    log_df = view_attendance_log()
    classes_df = load_classes()
    
    # Create filters
    col1, col2 = st.columns(2)
    
    # Class filter
    class_names = ["All Classes"] + list(classes_df["Class Name"].unique())
    selected_class = col1.selectbox("Filter by Class", class_names)
    
    # Date filter
    if not log_df.empty:
        # Convert Date column to datetime if it's string
        if isinstance(log_df["Date"].iloc[0], str):
            log_df["Date"] = pd.to_datetime(log_df["Date"])
        
        dates = ["All Dates"] + sorted(log_df["Date"].dt.date.unique().tolist())
        selected_date = col2.selectbox("Filter by Date", dates)
    else:
        selected_date = "All Dates"
    
    # Apply filters
    filtered_df = log_df.copy()
    
    if selected_class != "All Classes":
        filtered_df = filtered_df[filtered_df["Class"] == selected_class]
    
    if selected_date != "All Dates":
        filtered_df = filtered_df[filtered_df["Date"] == str(selected_date)]
    
    # Display filtered data
    if filtered_df.empty:
        st.info("No attendance records found for the selected filters.")
    else:
        # Add a count of records
        st.success(f"Found {len(filtered_df)} attendance records")
        st.dataframe(filtered_df)
        
        # Add an option to download the filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Filtered Data",
            csv,
            "attendance_filtered.csv",
            "text/csv",
            key='download-csv'
        )

elif choice == "Manage Students":
    st.header("🧑‍🎓 Registered Students")
    db = load_database()
    if db["names"]:
        df = pd.DataFrame({"Name": db["names"]})
        st.dataframe(df)

        name_to_delete = st.selectbox("Delete Student", db["names"])
        if st.button("Delete"):
            index = db["names"].index(name_to_delete)
            db["names"].pop(index)
            db["embeddings"].pop(index)
            save_database(db)
            st.success(f"Deleted student: {name_to_delete}")
    else:
        st.info("No students registered.")
