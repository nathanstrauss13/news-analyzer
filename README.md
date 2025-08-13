# News Analyzer

A web application for analyzing news articles based on search terms.

## Running the Application

The application runs on port 5008 by default. To start the application:

```bash
cd news-analyzer
python app.py
```

## Pushing Changes to GitHub

To push changes to the GitHub repository:

### First-time Setup

1. Ensure you have SSH keys set up for GitHub:
   ```bash
   # Generate a new SSH key
   ssh-keygen -t ed25519 -C "your-email@example.com"
   
   # Start the SSH agent
   eval "$(ssh-agent -s)"
   
   # Add your SSH key to the agent
   ssh-add ~/.ssh/id_ed25519
   
   # Display your public key to add to GitHub
   cat ~/.ssh/id_ed25519.pub
   ```

2. Add the SSH key to your GitHub account:
   - Go to GitHub.com and log in
   - Click your profile picture → Settings
   - In the sidebar, click "SSH and GPG keys"
   - Click "New SSH key"
   - Add a title and paste your key
   - Click "Add SSH key"

3. Test your SSH connection:
   ```bash
   ssh -T git@github.com
   ```

### Pushing Changes

Once SSH is set up, you can push changes with these commands:

```bash
# Navigate to the project directory
cd news-analyzer

# Add all changes
git add .

# Commit changes with a descriptive message
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

If you encounter issues with divergent branches:

```bash
# Configure git to use merge strategy
git config pull.rebase false

# Pull remote changes
git pull origin main

# Push your changes
git push origin main
```

## Repository Information

- GitHub Repository: https://github.com/nathanstrauss13/news-analyzer
