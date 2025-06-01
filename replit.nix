{ pkgs }: {
  deps = [
    pkgs.nodejs_20
    pkgs.pnpm
  ];
  shellHook = 
    export NODE_OPTIONS="--no-warnings"
    pnpm install --frozen-lockfile
  ;
}
