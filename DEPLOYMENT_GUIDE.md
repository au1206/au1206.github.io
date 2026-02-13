# Deployment Guide

## What's Been Completed ✅

### Phase 1: Repository Setup
- ✅ Hugo v0.155.3 installed
- ✅ New Hugo site created at `/Users/auppal/dev/akshayuppal.github.io`
- ✅ Git repository initialized
- ✅ PaperMod theme installed as submodule

### Phase 2: Configuration
- ✅ Hugo configuration (`hugo.toml`) created with:
  - Site metadata and branding
  - Dark theme by default
  - Profile mode with navigation buttons
  - Social media links
  - Google Analytics (G-3B9WV5JCK6)
  - Search functionality enabled
  - Pagination configured
  - Markup settings for HTML support

### Phase 3: Content Migration
- ✅ All 14 blog posts migrated from Jekyll to Hugo format
- ✅ 13 PDF files copied (34MB of annotated papers)
- ✅ 18 images copied
- ✅ About page created
- ✅ Search page created
- ✅ ads.txt copied

### Phase 4: Integrations
- ✅ Google Analytics configured
- ✅ Google AdSense script added (`layouts/partials/extend_head.html`)
- ✅ ads.txt file in place

### Phase 5: GitHub Actions
- ✅ Deployment workflow created (`.github/workflows/deploy.yml`)
- ✅ Configured for automatic deployment on push to `main` branch

### Phase 6: Testing
- ✅ Local Hugo server tested successfully
- ✅ All posts render correctly
- ✅ No build errors
- ✅ Build time: ~97ms (vs 2+ minutes with Jekyll!)
- ✅ 35 pages generated
- ✅ All assets accessible

### Phase 7: Git Setup
- ✅ .gitignore created
- ✅ README.md created
- ✅ Initial commit made with all files

## What's Next 🚀

### Step 1: Create GitHub Repository

You need to create the GitHub repository manually. You have two options:

#### Option A: Using GitHub CLI (Recommended)

1. Authenticate with GitHub:
```bash
gh auth login
```

2. Create the repository:
```bash
cd /Users/auppal/dev/akshayuppal.github.io
gh repo create akshayuppal.github.io --public --source=. --remote=origin --push
```

#### Option B: Manual Creation

1. Go to https://github.com/new
2. Repository name: `akshayuppal.github.io` (exactly this format for GitHub Pages)
3. Set to **Public**
4. **Do NOT** initialize with README, .gitignore, or license
5. Click "Create repository"
6. Run these commands:
```bash
cd /Users/auppal/dev/akshayuppal.github.io
git remote add origin https://github.com/akshayuppal/akshayuppal.github.io.git
git push -u origin main
```

### Step 2: Configure GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages** (in left sidebar)
3. Under "Build and deployment":
   - Source: Select **GitHub Actions**
   - Save
4. Wait 2-3 minutes for the first deployment

### Step 3: Verify Deployment

1. Go to the **Actions** tab in your GitHub repository
2. You should see a workflow run for "Deploy Hugo site to GitHub Pages"
3. Wait for it to complete (usually 1-2 minutes)
4. Visit `https://akshayuppal.github.io`

### Step 4: Test Your Site

Check these features:
- [ ] Homepage loads with your profile
- [ ] All 14 posts are visible
- [ ] PDFs embed correctly
- [ ] Images display properly
- [ ] Search works
- [ ] Categories page works
- [ ] Tags page works
- [ ] About page loads
- [ ] Dark theme is active
- [ ] Social media links work

## Ongoing Workflow (After Deployment)

### Adding a New Post

1. Create file: `content/posts/YYYY-MM-DD-title.md`
2. Add frontmatter:
```yaml
---
title: "Your Post Title"
date: 2026-02-14T10:00:00-04:00
categories: ["Annotated Paper"]
tags: ["NLP", "Transformers"]
draft: false
ShowToc: true
---
```
3. Write your content
4. For PDFs:
   - Place in `static/pdfs/your-paper.pdf`
   - Embed: `<embed src="/pdfs/your-paper.pdf" width="1000px" height="2100px" />`
5. Commit and push:
```bash
git add content/posts/YYYY-MM-DD-title.md
git add static/pdfs/your-paper.pdf  # if adding PDF
git commit -m "Add post: Your Post Title"
git push
```
6. Site auto-deploys in 2-3 minutes

### Local Testing

```bash
cd /Users/auppal/dev/akshayuppal.github.io
hugo server
# Visit http://localhost:1313
```

## Post-Launch Tasks

### Update Social Media Links
- [ ] LinkedIn profile: Update website to `https://akshayuppal.github.io`
- [ ] Twitter bio: Update link
- [ ] GitHub profile: Update website

### Handle Old Repository (au1206.github.io)

After confirming the new site works (wait 1-2 weeks):

**Option 1: Add Redirect Notice**
```bash
cd /Users/auppal/dev/au1206.github.io
echo "# MOVED

This blog has moved to [akshayuppal.github.io](https://akshayuppal.github.io)

All content has been migrated to the new site." > README.md
git add README.md
git commit -m "Add redirect notice to new site"
git push
```
Then archive the repository: Settings → General → Archive this repository

**Option 2: Delete** (after 2+ weeks of testing)
- Settings → General → Danger Zone → Delete this repository

## Troubleshooting

### Build Fails on GitHub Actions

Check the Actions tab for error messages. Common issues:
- Submodule not initialized: Make sure `.gitmodules` is committed
- Theme missing: Verify PaperMod submodule is present

### PDFs Not Displaying

- Ensure PDF is in `static/pdfs/` directory
- Path in markdown should be `/pdfs/filename.pdf` (not `/static/pdfs/`)
- Commit and push the PDF file

### Images Not Showing

- Ensure image is in `static/images/` directory
- Path in markdown should be `/images/filename.png`
- Commit and push the image file

### Site Not Updating

- Check Actions tab for deployment status
- Clear browser cache (Cmd+Shift+R)
- Wait 2-3 minutes after push

## Performance Comparison

| Metric | Jekyll (Old) | Hugo (New) |
|--------|-------------|------------|
| Build time | ~2 minutes | ~97ms |
| Page load | ~2 seconds | <500ms |
| Hot reload | No | Yes (instant) |
| Local dev | Slow | Fast |

## Support

If you encounter issues:
1. Check the Hugo documentation: https://gohugo.io/documentation/
2. PaperMod theme docs: https://github.com/adityatelange/hugo-PaperMod
3. GitHub Pages docs: https://docs.github.com/pages

## Summary

Your blog is ready to deploy! Just create the GitHub repository and push. The site will automatically build and deploy via GitHub Actions. You'll have a modern, fast, and easy-to-maintain ML blog at `akshayuppal.github.io`.

**Next command to run:**
```bash
gh auth login  # If not already authenticated
cd /Users/auppal/dev/akshayuppal.github.io
gh repo create akshayuppal.github.io --public --source=. --remote=origin --push
```

Then visit https://github.com/akshayuppal/akshayuppal.github.io/actions to watch your site deploy!
