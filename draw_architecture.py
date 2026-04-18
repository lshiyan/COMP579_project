"""Generate architecture diagram for the COMP579 Chameleon RL project."""
import subprocess, textwrap

dot_source = textwrap.dedent(r"""
digraph G {
    rankdir=TB;
    fontname="Helvetica";
    node [fontname="Helvetica", fontsize=11, style=filled, shape=box, penwidth=1.2];
    edge [fontname="Helvetica", fontsize=9];
    compound=true;
    newrank=true;
    nodesep=0.6;
    ranksep=0.7;

    // ── Title ──
    labelloc="t";
    label=<<B>COMP579 — RL for The Chameleon Game</B><BR/><FONT POINT-SIZE="10">Training Architecture &amp; Data Flow</FONT>>;
    fontsize=16;

    // ══════════════════════════════════════════
    //  GAME ENVIRONMENT
    // ══════════════════════════════════════════
    subgraph cluster_game {
        label=<<B>Game Environment (Chameleon POMDP)</B>>;
        style="rounded,filled"; fillcolor="#e8f0fe"; color="#4285f4"; penwidth=1.5;

        game_state [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
            <TR><TD><B>Hidden State</B></TD></TR>
            <TR><TD ALIGN="LEFT">• Secret word (unknown to chameleon)</TD></TR>
            <TR><TD ALIGN="LEFT">• Chameleon identity (unknown to others)</TD></TR>
            <TR><TD ALIGN="LEFT">• Topic &amp; word list (known to all)</TD></TR>
            </TABLE>
        >, fillcolor="#d2e3fc", shape=note];

        phases [label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
            <TR>
              <TD BGCOLOR="#c6efce"><B>Phase 1</B><BR/>Give Clues<BR/><FONT POINT-SIZE="8">(sequential, 1+ rounds)</FONT></TD>
              <TD BGCOLOR="#fff2cc"><B>Phase 2</B><BR/>Vote / Accuse<BR/><FONT POINT-SIZE="8">(sequential, private)</FONT></TD>
              <TD BGCOLOR="#fce4ec"><B>Phase 3</B><BR/>Guess Word<BR/><FONT POINT-SIZE="8">(chameleon only)</FONT></TD>
            </TR>
            </TABLE>
        >, fillcolor=white, shape=plaintext];

        obs [label="Observations\n(message history\nfiltered by visibility)", fillcolor="#bbdefb", shape=oval];
        rewards_node [label="Terminal Rewards\n+1 / 0 per player", fillcolor="#bbdefb", shape=oval];

        game_state -> phases [style=invis];
        phases -> obs [label="  per-player view"];
        phases -> rewards_node [label="  game over", style=dashed];
    }

    // ══════════════════════════════════════════
    //  PLAYER (LLM + BELIEF)
    // ══════════════════════════════════════════
    subgraph cluster_player {
        label=<<B>Player Agent</B>>;
        style="rounded,filled"; fillcolor="#fef7e0"; color="#f9ab00"; penwidth=1.5;

        subgraph cluster_llm {
            label=<<B>LLM Policy</B><BR/><FONT POINT-SIZE="9">Qwen 2.5 (1.5B / 7B)</FONT>>;
            style="rounded,filled"; fillcolor="#fff9c4"; color="#f9ab00";
            llm [label="Transformer\n(trainable weights)", fillcolor="#fff176"];
            ref_model [label="Reference Model\n(frozen copy)", fillcolor="#e0e0e0"];
        }

        subgraph cluster_belief {
            label=<<B>Belief Network</B>>;
            style="rounded,filled"; fillcolor="#e8f5e9"; color="#34a853";

            sent_enc [label="Sentence Encoder\n(all-MiniLM-L6-v2)\n384-dim embeddings", fillcolor="#a5d6a7"];
            gru [label="GRU Cell\n(belief updater)\nhidden_size=512", fillcolor="#81c784"];
            emb [label="Speaker + Role\nEmbeddings", fillcolor="#a5d6a7"];

            subgraph cluster_heads {
                label="Belief Heads"; style="rounded,filled"; fillcolor="#c8e6c9"; color="#66bb6a";
                player_head [label="Player Head\n(Linear → softmax)\n\"Who is chameleon?\"", fillcolor="#e8f5e9"];
                word_head [label="Word Head\n(Linear → softmax)\n\"What is secret word?\"", fillcolor="#e8f5e9"];
            }

            sent_enc -> gru;
            emb -> gru;
            gru -> player_head;
            gru -> word_head;
        }
    }

    // ══════════════════════════════════════════
    //  GRPO TRAINING LOOP
    // ══════════════════════════════════════════
    subgraph cluster_grpo {
        label=<<B>GRPO Training Loop</B><BR/><FONT POINT-SIZE="9">(non-chameleon clue phase only)</FONT>>;
        style="rounded,filled"; fillcolor="#fce4ec"; color="#ea4335"; penwidth=1.5;

        gen [label="① Generate K clues\n(K=3 candidates)", fillcolor="#f8bbd0"];
        score [label="② Score each clue\nvia belief reward", fillcolor="#f8bbd0"];
        adv [label="③ Compute advantages\n(normalize rewards)", fillcolor="#f8bbd0"];
        update [label="④ GRPO update\n(clipped ratio × advantage\n+ β · KL from ref model)", fillcolor="#ef9a9a"];
        best [label="⑤ Submit best clue\nto game", fillcolor="#f8bbd0"];

        gen -> score -> adv -> update -> best;
    }

    // ══════════════════════════════════════════
    //  BELIEF TRAINING
    // ══════════════════════════════════════════
    subgraph cluster_belief_train {
        label=<<B>Belief Training</B>>;
        style="rounded,filled"; fillcolor="#e8f5e9"; color="#34a853"; penwidth=1.5;

        ce_loss [label="Cross-Entropy Loss\nvs ground truth\n(true chameleon / true word)", fillcolor="#a5d6a7"];
        belief_opt [label="Belief Optimizer\n(Adam, lr=1e-3)\nupdates GRU + heads", fillcolor="#81c784"];

        ce_loss -> belief_opt;
    }

    // ══════════════════════════════════════════
    //  REWARD SIGNAL
    // ══════════════════════════════════════════
    subgraph cluster_reward {
        label=<<B>Belief Reward</B>>;
        style="rounded,filled"; fillcolor="#f3e5f5"; color="#9c27b0"; penwidth=1.5;

        reward_fn [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
            <TR><TD><B>r = α · suspicion + (1-α) · word_leak</B></TD></TR>
            <TR><TD ALIGN="LEFT">suspicion: did others' belief that<BR/>I'm chameleon <I>decrease</I>?</TD></TR>
            <TR><TD ALIGN="LEFT">word_leak: did chameleon's belief<BR/>about secret word <I>not increase</I>?</TD></TR>
            </TABLE>
        >, fillcolor="#e1bee7", shape=note];
    }

    // ══════════════════════════════════════════
    //  EDGES (data flow)
    // ══════════════════════════════════════════

    // Game → Player
    obs -> llm [label="  prompt\n  (messages)", lhead=cluster_player, color="#4285f4", penwidth=1.3];
    obs -> sent_enc [label="  clue text  ", color="#34a853", penwidth=1.3];

    // Player → GRPO
    llm -> gen [label="  sample\n  responses", lhead=cluster_grpo, color="#ea4335", penwidth=1.3];

    // Belief → Reward
    player_head -> reward_fn [label="  prior/post\n  beliefs", color="#9c27b0", penwidth=1.3, ltail=cluster_heads];
    word_head -> reward_fn [color="#9c27b0", penwidth=1.3, style=invis];

    // Reward → GRPO
    reward_fn -> score [label="  per-clue\n  reward", color="#9c27b0", penwidth=1.3];

    // GRPO → LLM (weight update)
    update -> llm [label="  policy\n  gradient", color="#ea4335", penwidth=1.3, style=bold];

    // Ref model → GRPO
    ref_model -> update [label="  KL anchor", color="#757575", style=dashed];

    // Best clue → Game
    best -> phases [label="  action\n  submitted", color="#4285f4", penwidth=1.3];

    // Belief training
    gru -> ce_loss [label="  belief\n  logits", color="#34a853", penwidth=1.3, ltail=cluster_belief];
    belief_opt -> gru [label="  gradient\n  update", color="#34a853", penwidth=1.3, style=bold, lhead=cluster_belief];

    // ══════════════════════════════════════════
    //  LEGEND
    // ══════════════════════════════════════════
    subgraph cluster_legend {
        label=<<B>Legend</B>>;
        style="rounded,filled"; fillcolor="#f5f5f5"; color="#9e9e9e";
        node [shape=plaintext, fillcolor="#f5f5f5"];

        legend [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
            <TR><TD BGCOLOR="#d2e3fc" WIDTH="18"> </TD><TD ALIGN="LEFT">Game Environment</TD>
                <TD>  </TD>
                <TD BGCOLOR="#fff176" WIDTH="18"> </TD><TD ALIGN="LEFT">LLM Policy (trainable)</TD></TR>
            <TR><TD BGCOLOR="#81c784" WIDTH="18"> </TD><TD ALIGN="LEFT">Belief Network (trainable)</TD>
                <TD>  </TD>
                <TD BGCOLOR="#e1bee7" WIDTH="18"> </TD><TD ALIGN="LEFT">Reward Signal</TD></TR>
            <TR><TD BGCOLOR="#ef9a9a" WIDTH="18"> </TD><TD ALIGN="LEFT">GRPO Training</TD>
                <TD>  </TD>
                <TD BGCOLOR="#e0e0e0" WIDTH="18"> </TD><TD ALIGN="LEFT">Frozen / Reference</TD></TR>
            </TABLE>
        >];
    }

    // Force ordering
    { rank=same; game_state; phases; }
}
""")

with open("/tmp/arch.dot", "w") as f:
    f.write(dot_source)

subprocess.run(
    ["dot", "-Tpng", "-Gdpi=150", "/tmp/arch.dot",
     "-o", "/home/anthony/Projects/COMP579_project/architecture.png"],
    check=True,
)
print("Written to architecture.png")
