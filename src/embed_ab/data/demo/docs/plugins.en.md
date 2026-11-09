Plugins are small folders containing a manifest and a script. Drop one into the plugins directory and restart the app to load it.

Every plugin declares the permissions it needs in its manifest. The app asks you to approve network or file access the first time it is used.

The scripting API exposes notes, tags and the editor. Use app.notes.create(title, body) to add a note from a plugin.

Plugins run in a sandbox with a 5 second time limit per call. Long tasks should be split into chunks or scheduled as background jobs.

To publish a plugin, add a release on its repository and submit the repository name to the community list. A reviewer checks the manifest before it appears.

Crashing plugins are disabled automatically after three failures in a row. Re-enable them from the plugin list once the problem is fixed.

Plugin settings are stored per vault in a small JSON file. Delete the file to reset a plugin to its defaults.

Themes are plugins that only contain CSS. They load without a script and never need permissions.
