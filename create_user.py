#!/usr/bin/env python3
"""
Script to create Airflow users programmatically
"""
from airflow.providers.fab.auth_manager.api.auth.backend.session import SessionAuthManager
from airflow.providers.fab.auth_manager.models.user import User
from airflow.utils.db import get_session

def create_airflow_user(username, email, first_name, last_name, password, role="User"):
    """Create a new Airflow user"""
    auth_manager = SessionAuthManager()
    
    with get_session() as session:
        # Create user
        user = User()
        user.username = username
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.password = auth_manager.hash_password(password)
        user.active = True
        
        # Add role
        role_obj = auth_manager.find_role(role)
        if role_obj:
            user.roles = [role_obj]
        
        session.add(user)
        session.commit()
        print(f"✅ User '{username}' created successfully!")

if __name__ == "__main__":
    # Example: Create a new user
    create_airflow_user(
        username="analyst",
        email="analyst@company.com", 
        first_name="Data",
        last_name="Analyst",
        password="secure_password123",
        role="User"
    )
