// Run with Node >=22 and --experimental-strip-types.
// Import the actual production TypeScript simulator, not a rewritten copy.
import { pathToFileURL } from "node:url";
const source = process.env.GEOMETRY_TS_CORE;
if (!source) throw Error("GEOMETRY_TS_CORE must point to portfolio/app/lib/geometry-core.ts");
const { initial, update, STEP } = await import(pathToFileURL(source).href);

function snapshot(g) {
  return {
    seed:g.seed, x:g.x,y:g.y,vx:g.vx,vy:g.vy,t:g.t,
    score:g.score,spawn:g.spawn,dead:g.dead,nearCount:g.nearCount,
    enemies:g.enemies.map(e=>({
      x:e.x,y:e.y,vx:e.vx,vy:e.vy,r:e.r,kind:e.kind,angle:e.angle
    }))
  };
}
const cases = [
  {seed:1, mode:0}, {seed:42,mode:1}, {seed:271828,mode:2},
  {seed:4294967295,mode:3}, {seed:20261023,mode:4}
];
const actions=[[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]];
const output=[];
for(const test of cases) {
  const g=initial(test.seed), states=[];
  for(let i=0;i<12000;i++) {
    const action=actions[(Math.floor(i/6)+test.mode*3)%actions.length];
    update(g,STEP,...action);
    if(i%97===0 || g.dead) states.push({tick:i+1,state:snapshot(g)});
    if(g.dead) break;
  }
  output.push({seed:test.seed,mode:test.mode,states});
}
process.stdout.write(JSON.stringify(output));
