## GitHub Copilot Chat

- Extension: 0.46.2 (prod)
- VS Code: 1.118.1 (034f571df509819cc10b0c8129f66ef77a542f0e)
- OS: linux 6.6.87.2-microsoft-standard-WSL2 x64
- Remote Name: wsl
- Extension Kind: Workspace
- GitHub Account: TKevin0v0

## Network

User Settings:
```json
  "http.systemCertificatesNode": true,
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.205.243.168 (73 ms)
- DNS ipv6 Lookup: Error (8 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: http://127.0.0.1:7897 (1 ms)
- Proxy Connection: Error (3 ms): connect ECONNREFUSED 127.0.0.1:7897
- Electron fetch: Unavailable
- Node.js https: Error (11 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7897
	at PacProxyAgent.<anonymous> (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
- Node.js fetch (configured): Error (14 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async t._fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:5229)
	at async t.fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:4541)
	at async u (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5357:186)
	at async Pg._executeContributedCommand (file:///home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/out/vs/workbench/api/node/extensionHostProcess.js:503:48675)
  Error: connect ECONNREFUSED 127.0.0.1:7897
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
  	at TCPConnectWrap.callbackTrampoline (node:internal/async_hooks:130:17)

Connecting to https://api.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.113.21 (63 ms)
- DNS ipv6 Lookup: Error (73 ms): getaddrinfo ENOTFOUND api.githubcopilot.com
- Proxy URL: http://127.0.0.1:7897 (0 ms)
- Proxy Connection: Error (1 ms): connect ECONNREFUSED 127.0.0.1:7897
- Electron fetch: Unavailable
- Node.js https: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7897
	at PacProxyAgent.<anonymous> (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
- Node.js fetch (configured): Error (12 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async t._fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:5229)
	at async t.fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:4541)
	at async u (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5357:186)
	at async Pg._executeContributedCommand (file:///home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/out/vs/workbench/api/node/extensionHostProcess.js:503:48675)
  Error: connect ECONNREFUSED 127.0.0.1:7897
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
  	at TCPConnectWrap.callbackTrampoline (node:internal/async_hooks:130:17)

Connecting to https://copilot-proxy.githubusercontent.com/_ping:
- DNS ipv4 Lookup: 4.249.131.160 (11 ms)
- DNS ipv6 Lookup: Error (887 ms): getaddrinfo ENOTFOUND copilot-proxy.githubusercontent.com
- Proxy URL: http://127.0.0.1:7897 (51 ms)
- Proxy Connection: Error (1 ms): connect ECONNREFUSED 127.0.0.1:7897
- Electron fetch: Unavailable
- Node.js https: Error (3 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7897
	at PacProxyAgent.<anonymous> (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
- Node.js fetch (configured): Error (9 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async t._fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:5229)
	at async t.fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:4541)
	at async u (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5357:186)
	at async Pg._executeContributedCommand (file:///home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/out/vs/workbench/api/node/extensionHostProcess.js:503:48675)
  Error: connect ECONNREFUSED 127.0.0.1:7897
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
  	at TCPConnectWrap.callbackTrampoline (node:internal/async_hooks:130:17)

Connecting to https://mobile.events.data.microsoft.com: Error (8 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async t._fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:5229)
	at async t.fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:4541)
	at async u (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5362:137)
	at async Pg._executeContributedCommand (file:///home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/out/vs/workbench/api/node/extensionHostProcess.js:503:48675)
  Error: connect ECONNREFUSED 127.0.0.1:7897
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
  	at TCPConnectWrap.callbackTrampoline (node:internal/async_hooks:130:17)
Connecting to https://dc.services.visualstudio.com: Error (66 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async t._fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:5229)
	at async t.fetch (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5325:4541)
	at async u (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/extensions/copilot/dist/extension.js:5362:137)
	at async Pg._executeContributedCommand (file:///home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/out/vs/workbench/api/node/extensionHostProcess.js:503:48675)
  Error: connect ECONNREFUSED 127.0.0.1:7897
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
  	at TCPConnectWrap.callbackTrampoline (node:internal/async_hooks:130:17)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: Error (3 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7897
	at PacProxyAgent.<anonymous> (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7897
	at PacProxyAgent.<anonymous> (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
Connecting to https://default.exp-tas.com: Error (3 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7897
	at PacProxyAgent.<anonymous> (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/home/tkevin/.vscode-server/bin/034f571df509819cc10b0c8129f66ef77a542f0e/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)

Number of system certificates: 399

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).