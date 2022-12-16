# Cheatsheet for app deployment

There are some things that it is worth getting right first time, as it is difficult or tedious to change them later. These include:
+ The name of the landing page script, e.g. `Introduction.py`. Changing this seems to require deleting and re-deploying the app.
+ The name and location of the code repository, including the owner on GitHub (a person or an organisation). This seems to continue working seamlessly but I don't like the look of it.

Although the URL of the app can be changed easily, the old URL will not redirect to the new location. So settle on the final URL before you spread the link far and wide.

### Custom URL 

To change the URL:
1. Go to the list of your apps, `https://share.streamlit.io/`
2. Click the "hamburger" menu next to the app you want to change
3. Click "Settings"
4. Under the "General" tab, follow the instructions to change URL.
5. Press the "Enter" key when you're done if the "Save" button is playing up.


### Requirements file

The documentation says that a conda environment `.yml` file works with Streamlit. I've been unable to prove this. Instead use a `pip` requirements file like the one in the template.

### Streamlit badge

To include a badge in your GitHub readme, include the following text:

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your url here)

## Repository ownership

You can launch an app from a repository owned by yourself (e.g. `aselaws/health_streamlit_template/` is owned by me, `aselaws`) or by an organisation (e.g. `samuel-book/stroke_outcome_app/` is owned by the organisation `samuel-book`). 

If the repo belongs to an organisation, the app must be deployed by an owner of the organisation. Otherwise Streamlit gives the following error message:

> "You are not authorized to perform the requested action: You must be a Github Administrator for the repo to deploy an app. You currently have push access."

### Deploying an app owned by an organisation

The default Streamlit page `https://share.streamlit.io/` shows all of the apps owned by you personally. In the top right, you can click your name to switch workspaces where there is one workspace per GitHub organisation. To deploy a new app, you must be in the workspace that matches the GitHub profile or organisation that owns the repository.

### Problem: Workspace not appearing

This can happen when you join an organisation on GitHub _after_ linking your Streamlit and GitHub accounts, so Streamlit doesn't automatically have permission to access the "new" organisation.

To fix this, try following these steps:
1. On GitHub, go to this page: `Your profile (not the organisation's) > Settings > Applications > Authorized OAuth Apps`
2. Click on the word "Streamlit" to see sections for "Permissions" and "Organization access". In the latter you can allow Streamlit to access specific organisations.

This should work without you needing to be an owner of the organisation. Apparently the organisation must contain at least one public repository for this to work as expected, but I haven't tested this.

### Transfer ownership of repository

> This should be avoided.

GitHub has an option to "transfer ownership" of a repository. Doing this has strange effects on an existing app. 
+ The app continues to work as normal, but I haven't tested how long that effect lasts or whether it will update when new changes are pushed.
+ The app will disappear from the list of apps on the `https://share.streamlit.io/` page. It does not materialise in the list of apps for any other workspace. At this point, the app still runs without problems. 

To reboot or delete the app, you must go to the app page while logged in to Streamlit and use the options in the "Manage app" menu in the corner.

After transferring ownership of a repo, I have decided to delete the old app and re-deploy the app from the new location.
Deleting and re-deploying an app does lose all of the analytics about how many viewers you've had and when. 

You can hide this blunder by changing the new URL to be the same as the old one. Nobody need know!